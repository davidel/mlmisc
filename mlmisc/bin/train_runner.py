import argparse
import operator
import os

import mlmisc.auto_module as mlam
import mlmisc.config as mlco
import mlmisc.core_utils as mlcu
import mlmisc.load_state_dict as mlsd
import mlmisc.torch_profiler as mltp
import mlmisc.trainer as mltr
import py_misc_utils.alog as alog
import py_misc_utils.app_main as pyam
import py_misc_utils.assert_checks as tas
import py_misc_utils.break_control as pybc
import py_misc_utils.core_utils as pycu
import py_misc_utils.gfs as gfs
import py_misc_utils.inspect_utils as pyiu
import py_misc_utils.module_utils as pymu
import py_misc_utils.utils as pyu
import torch
import torch.nn as nn

from . import base_setup as bs
from . import dataset as ds
from ..lrsched import wrapper as lrw


def create_profiler(prof_config):
  if prof_config is not None:
    pcfg = pyu.parse_dict(prof_config)
    alog.debug0(f'Profiler Config: {pcfg}')

    profiler = mltp.create_profiler(pcfg)
  else:
    profiler = mltp.NoopProfiler()

  return profiler


def create_stepfn(tprof):
  def stepfn(*args, **kwargs):
    tprof.step()

  return stepfn


def create_config(dataset):
  x, y = pycu.seqfirst(dataset)
  x_shape = tuple(getattr(x, 'shape', ()))
  y_shape = tuple(getattr(y, 'shape', ()))
  alog.info(f'Dataset: xshape={x_shape} yshape={y_shape}')

  config = dict(dataset=dataset, x_shape=x_shape, y_shape=y_shape)

  extra_arg = getattr(dataset, 'extra_arg', None)
  if extra_arg is not None:
    if (tokenizer := extra_arg('tokenizer')) is not None:
      for attr in ('vocab_size', 'bos_id', 'eos_id', 'unk_id'):
        if (attrfn := getattr(tokenizer, attr, None)) is not None:
          config[attr] = attrfn()

  alog.debug(f'Dataset Config: {config}')

  return config


def replace_args(model_args, model_kwargs, config):
  args, kwargs = [], dict()
  for arg in model_args:
    if isinstance(arg, str) and arg.startswith('$') and arg[1:] in config:
      arg = config[arg[1:]]
    args.append(arg)
  for name, arg in model_kwargs.items():
    if isinstance(arg, str) and arg.startswith('$') and arg[1:] in config:
      arg = config[arg[1:]]
    kwargs[name] = arg

  return args, kwargs


def load_model_state(source):
  if gfs.is_path(source):
    # Load straight from checkpoint.
    return mlcu.torch_load(source, weights_only=True)

  # Support loading from external model checkpoints (like from_pretrained() things
  # from Hugginface for example).
  model = mlco.create_object('Source Model', source)

  return model.state_dict()


def init_model(model, init_mappings):
  mappings = pyu.parse_config(init_mappings)

  for imap in pyu.as_sequence(mappings):
    source, maps = imap['source'], imap['maps']

    state = load_model_state(source)

    for name, param in model.named_parameters():
      sname = maps.get(name)
      if sname is not None:
        sparam = state.get(sname)
        tas.check_is_not_none(sparam, msg=f'Parameter \"{sname}\" not found ' \
                              f'in \"{path}\" checkpoint (required by \"{name}\")')

        param.data.copy_(getattr(sparam, 'data', sparam))


def create_model(args, trainer, dataset):
  module = pymu.import_module(args.model_path, modname='train_module')

  model, state = None, dict()
  if gfs.exists(args.checkpoint_path):
    if args.rebuild_model:
      alog.debug0(f'Loading raw state from {args.checkpoint_path}')
      state = trainer.load_raw_state(args.checkpoint_path)
      trainer.load_state(state)
    else:
      model, state = trainer.load_model(args.checkpoint_path,
                                        strict=args.strict)

  if model is None:
    model_function, *cmdline_args = args.extra_args

    model_args, model_kwargs = pyu.parse_args(cmdline_args)

    config = create_config(dataset)
    model_args, model_kwargs = replace_args(model_args, model_kwargs, config)

    alog.info(f'Model Args: {model_args}\nModel Kwargs: {model_kwargs}')

    model_ctor = operator.attrgetter(model_function)(module)

    model = mlam.create(model_ctor, *model_args, **model_kwargs)

    if args.init_mappings:
      init_model(model, args.init_mappings)

    model_state = trainer.model_state(state)
    if model_state is not None:
      mlsd.load_state_dict(model, mlam.purged_state(model_state), strict=args.strict)

  model = model.to(args.device)

  alog.info(f'Model Network:\n{model}')
  alog.info(f'Model Parameters:')
  for name, param in model.named_parameters():
    alog.info(f'  {name}\t{tuple(param.shape)}')
  alog.info(f'Model {pyiu.cname(model)} has {mlcu.count_params(model):.2e} parameters')

  return model, state


def main(args):
  bs.setup(args)

  train_dataset, test_dataset = ds.create_dataset(args)

  trainer = mltr.Trainer()

  model, state = create_model(args, trainer, train_dataset)

  optimizer = mlco.create_optimizer(model.parameters(), args.optimizer)
  if args.load_optim_state:
    trainer.load_aux_state(state, optimizer=optimizer)

  if args.lr_scheduler:
    scheduler = mlco.create_lr_scheduler(optimizer, args.lr_scheduler)

    if args.wrap_lr_scheduler:
      wrap_kwargs = pyu.parse_config(args.wrap_lr_scheduler)
      alog.info(f'Wrapping LR Scheduler: {wrap_kwargs}')
      scheduler = lrw.wrap(scheduler, **wrap_kwargs)

    if args.load_lrsched_state:
      trainer.load_aux_state(state, scheduler=scheduler)
  else:
    scheduler = None

  if args.amp_dtype and mlcu.is_cuda_device(args.device):
    alog.info(f'Enabling AMP scaler for device {args.device} and type {args.amp_dtype}')
    scaler = torch.amp.GradScaler()
    trainer.load_aux_state(state, scaler=scaler)
    amp_dtype = mlcu.torch_dtype(args.amp_dtype)
  else:
    scaler, amp_dtype = None, None

  del state

  tprof = create_profiler(args.profiler)
  tb_writer = mlcu.create_tb_writer(args.tb_path) if args.tb_path else mlcu.NoopTbWriter()

  with pybc.BreakControl() as bc, tprof:
    step_fn = create_stepfn(tprof)

    for e in range(args.num_epochs):
      trainer.train_epoch(model, optimizer, train_dataset, test_dataset, args.batch_size,
                          device=args.device,
                          scheduler=scheduler,
                          grad_clip=args.grad_clip,
                          val_time=args.val_time,
                          loss_logstep=args.loss_logstep,
                          val_logstep=args.val_logstep,
                          model_chkptstep=args.checkpoint_step,
                          checkpoint=pyu.comma_split(args.checkpoint),
                          model_path=args.checkpoint_path,
                          tb_writer=tb_writer,
                          num_workers=args.num_workers,
                          drop_last=args.drop_last,
                          should_stop=lambda: bc.hit(),
                          step_fn=step_fn,
                          scaler=scaler,
                          amp_dtype=amp_dtype)

      if args.show_cuda_memory and mlcu.is_cuda_device(args.device):
        alog.info(f'CUDA Memory:\n{torch.cuda.memory_summary(device=args.device)}')

      if bc.hit():
        break

  if tb_writer is not None:
    tb_writer.flush()
    tb_writer.close()


if __name__ == '__main__':
  # Do basic logging setup ... will be setup later once the modules are parsed.
  alog.basic_setup()

  parser = argparse.ArgumentParser(description='Model Trainer',
                                   formatter_class=argparse.ArgumentDefaultsHelpFormatter)

  bs.add_parser_arguments(parser)
  ds.add_parser_arguments(parser)

  parser.add_argument('--model_path', required=True,
                      help='The path containing the model definition')
  parser.add_argument('--checkpoint_path', required=True,
                      help='The path to be used to store the model checkpoint')
  parser.add_argument('--rebuild_model', action=argparse.BooleanOptionalAction, default=False,
                      help='Force model rebuild while loading parameters from existing ' \
                      f'checkpoint (if any)')
  parser.add_argument('--init_mappings',
                      help='The configuration JSON string describing how to copy data ' \
                      f'into the new model parameters ' \
                      f'(or the path to a YAML/JSON file containing such configuration)')
  parser.add_argument('--amp_dtype',
                      help='The type to be used with the AMP (Automatic Mixed Precision) module')
  parser.add_argument('--batch_size', type=int, default=32,
                      help='The batch size to be used')
  parser.add_argument('--num_epochs', type=int, default=1,
                      help='The number of epochs to train for')
  parser.add_argument('--loss_logstep', type=int, default=15,
                      help='The number of seconds between loss display')
  parser.add_argument('--val_logstep', type=int, default=600,
                      help='The number of seconds between validation runs')
  parser.add_argument('--val_time', type=int, default=60,
                      help='The maximum number of seconds to be spent in validation')
  parser.add_argument('--checkpoint_step', type=int, default=120,
                      help='The number of seconds between checkpoints')
  parser.add_argument('--grad_clip', type=float,
                      help='The gradient clipping value')
  parser.add_argument('--num_workers', type=int, default=max(1, os.cpu_count() // 2),
                      help='The number of workers to be used by the data loaders')
  parser.add_argument('--checkpoint', default='scheduler,scaler',
                      help='The objects to be saved within the checkpoint file')
  parser.add_argument('--optimizer', required=True,
                      help='The configuration for the optimizer (class_path:arg0,...,name0=value0,...)')
  parser.add_argument('--load_optim_state', action=argparse.BooleanOptionalAction, default=True,
                      help='Whether to load the optimizer state')
  parser.add_argument('--lr_scheduler',
                      help='The configuration for the learning rate scheduler ' \
                      '(class_path:arg0,...,name0=value0,...)')
  parser.add_argument('--wrap_lr_scheduler',
                      help='The configuration for the learning rate scheduler wrapper')
  parser.add_argument('--load_lrsched_state', action=argparse.BooleanOptionalAction, default=True,
                      help='Whether to load the learning rate scheduler state')
  parser.add_argument('--drop_last', action=argparse.BooleanOptionalAction, default=True,
                      help='Whether the last incomplete batch should be dropped or ' \
                      f'returned with a lower batch size')
  parser.add_argument('--tb_path',
                      help='The path of the Tensorboard logging folder, if required')
  parser.add_argument('--profiler',
                      help='The comma-separated name=value string to be used for the ' \
                      'configuration of the PyTorch profiler')
  parser.add_argument('--strict', default='true',
                      choices=tuple(mlsd.VALID_STRICTS.keys()),
                      help='Which strict mode to use when loading model state dictionaries')
  parser.add_argument('--show_cuda_memory', action=argparse.BooleanOptionalAction, default=False,
                      help='Whether to log the current CUDA memory usage')

  pyam.main(parser, main, rem_args='extra_args')


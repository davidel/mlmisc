import argparse
import operator
import os

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms.v2 as transforms

import mlmisc.auto_module as mlam
import mlmisc.config as mlco
import mlmisc.dataset_utils as mldu
import mlmisc.torch_profiler as mltp
import mlmisc.trainer as mltr
import mlmisc.utils as mlut
import py_misc_utils.alog as alog
import py_misc_utils.app_main as pyam
import py_misc_utils.assert_checks as tas
import py_misc_utils.break_control as pybc
import py_misc_utils.utils as pyu


def create_profiler(prof_config):
  if prof_config:
    pcfg = pyu.parse_dict(prof_config)
    alog.debug(f'Profiler Config: {pcfg}')

    profiler = mltp.create_profiler(pcfg)
  else:
    profiler = mltp.NoopProfiler()

  return profiler


def create_stepfn(tprof):
  def stepfn(*args, **kwargs):
    tprof.step()

  return stepfn


def _create_dataset(args):
  train_trans, test_trans = nn.Identity(), nn.Identity()
  if args.dataset_transform:
    with open(args.dataset_transform, mode='r') as dtf:
      tcode = dtf.read()

    gbls = globals().copy()
    exec(tcode, gbls)
    train_trans = gbls.get('TRAIN_TRANS', train_trans)
    test_trans = gbls.get('TEST_TRANS', test_trans)

  alog.info(f'Train Image Augmentation:\n{train_trans}')
  alog.info(f'Test Image Augmentation:\n{test_trans}')

  if args.dataset_selector:
    select_fn = mldu.keys_selector(pyu.comma_split(args.dataset_selector))
  else:
    select_fn = None

  dataset_kwargs = pyu.parse_dict(args.dataset_kwargs or '')

  dsets = mldu.create_dataset(args.dataset,
                              root=args.cache_dir,
                              select_fn=select_fn,
                              transform=dict(train=train_trans, test=test_trans),
                              dataset_kwargs=dataset_kwargs)

  train_dataset = dsets['train']
  test_dataset = dsets['test']

  alog.info(f'Train/Test Dataset samples = {len(train_dataset)}/{len(test_dataset)}')

  if args.show_images:
    mldu.show_images(train_dataset, args.show_images)

  return train_dataset, test_dataset


def _replace_args(model_args, model_kwargs, config):
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


def _create_model(args, trainer, dataset):
  module = pyu.import_module(args.model_path, modname='train_module')

  if os.path.exists(args.checkpoint_path):
    model, state = trainer.load_model(args.checkpoint_path,
                                      device=args.device,
                                      strict=args.strict)
  else:
    model_function, *cmdline_args = args.model_args

    model_args, model_kwargs = [], dict()
    for arg in cmdline_args:
      parts = [x.strip() for x in pyu.split_unquote(arg, '=', maxsplit=1)]
      if len(parts) == 2:
        model_kwargs[pyu.infer_value(parts[0], vtype=str)] = pyu.infer_value(parts[1])
      else:
        model_args.append(pyu.infer_value(parts[0]))

    x, y = dataset[0]
    x_shape = tuple(getattr(x, 'shape', ()))
    y_shape = tuple(getattr(y, 'shape', ()))
    config = dict(dataset=dataset, x_shape=x_shape, y_shape=y_shape)

    model_args, model_kwargs = _replace_args(model_args, model_kwargs, config)

    model_ctor = operator.attrgetter(model_function)(module)

    model = mlam.create(model_ctor, *model_args, **model_kwargs)

    model = model.to(args.device)
    state = dict()

  alog.info(f'Model Network:\n{model}')
  alog.info(f'Model Parameters:')
  for name, param in model.named_parameters():
    alog.info(f'  {name}\t{tuple(param.shape)}')
  alog.info(f'Model {pyu.cname(model)} has {mlut.count_params(model):.2e} parameters')

  return model, state


def _common_main(args):
  if args.seed is not None:
    mlut.randseed(args.seed)
  if args.cpu_num_threads is not None:
    torch.set_num_threads(args.cpu_num_threads)
  if args.device is None:
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  else:
    args.device = torch.device(args.device)


def _main(args):
  _common_main(args)

  train_dataset, test_dataset = _create_dataset(args)

  trainer = mltr.Trainer()

  model, state = _create_model(args, trainer, train_dataset)

  optimizer = mlco.create_optimizer(model.parameters(), args.optimizer)
  if args.load_optim_state:
    trainer.load_aux_state(state, optimizer=optimizer)

  if args.lr_scheduler:
    scheduler = mlco.create_lr_scheduler(optimizer, args.lr_scheduler)
    if args.load_lrsched_state:
      trainer.load_aux_state(state, scheduler=scheduler)
  else:
    scheduler = None

  if args.amp_dtype and args.device.type == 'cuda':
    alog.info(f'Enabling AMP scaler for device {args.device} and type {args.amp_dtype}')
    scaler = torch.amp.GradScaler()
    trainer.load_aux_state(state, scaler=scaler)
    amp_dtype = mlut.torch_dtype(args.amp_dtype)
  else:
    scaler, amp_dtype = None, None

  del state

  tprof = create_profiler(args.profiler)

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
                          num_workers=args.num_workers,
                          should_stop=lambda: bc.hit(),
                          step_fn=step_fn,
                          scaler=scaler,
                          amp_dtype=amp_dtype)

      if bc.hit():
        break


if __name__ == '__main__':
  # Do basic logging setup ... will be setup later once the modules are parsed.
  alog.basic_setup()

  parser = argparse.ArgumentParser(description='Model Trainer',
                                   formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument('--model_path', required=True,
                      help='The path containing the model definition')
  parser.add_argument('--cache_dir',
                      default=os.path.join(os.getenv('HOME', '.'), '.cache'),
                      help='The folder used to store cached files')
  parser.add_argument('--device',
                      help='The device to be used')
  parser.add_argument('--seed', type=int,
                      help='The seed for the random number generators')
  parser.add_argument('--cpu_num_threads', type=int,
                      help='The number of threads to dedicate to the PyTorch CPU device')
  parser.add_argument('--checkpoint_path', required=True,
                      help='The path to be used to store the model checkpoint')
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
  parser.add_argument('--dataset', required=True,
                      help='The name of the dataset to be used')
  parser.add_argument('--dataset_transform',
                      help='The path to the Python file containing the TRAIN_TRANS ' \
                      'and TEST_TRANS dataset transformations')
  parser.add_argument('--dataset_selector',
                      help='The comma-separated keys to be used to select the dataset items')
  parser.add_argument('--dataset_kwargs',
                      help='The comma-separated key=value arguments for the dataset constructor')
  parser.add_argument('--show_images', type=int,
                      help='The number of sample images to be shown before starting training')
  parser.add_argument('--optimizer', required=True,
                      help='The configuration for the optimizer (class_path:arg0,...,name0=value0,...)')
  parser.add_argument('--load_optim_state', action=argparse.BooleanOptionalAction, default=True,
                      help='Whether to load the optimizer state')
  parser.add_argument('--lr_scheduler',
                      help='The configuration for the learning rate scheduler ' \
                      '(class_path:arg0,...,name0=value0,...)')
  parser.add_argument('--load_lrsched_state', action=argparse.BooleanOptionalAction, default=True,
                      help='Whether to load the learning rate scheduler state')
  parser.add_argument('--profiler',
                      help='The comma-separated name=value string to be used for the ' \
                      'configuration of the PyTorch profiler')
  parser.add_argument('--strict', action=argparse.BooleanOptionalAction, default=True,
                      help='Use strict mode when loading model state dictionaries')

  pyam.main(parser, _main, rem_args='model_args')


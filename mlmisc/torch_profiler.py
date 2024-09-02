import os

from py_misc_utils import alog
from py_misc_utils import assert_checks as tas
from py_misc_utils import utils as pyu
import torch


def _make_trace_handler(args):
  # https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html
  # https://pytorch.org/docs/stable/profiler.html
  row_limit = pyu.dget(args, 'prof.row_limit', 10)
  sort_by = pyu.dget(args, 'prof.sort_by', '')
  if not sort_by:
    sort_by = 'self_cuda_time_total' if torch.cuda.is_available() else 'cpu_time_total'
  traces_path = pyu.dget(args, 'prof.traces_path', '')
  stacks_path = pyu.dget(args, 'prof.stacks_path', '')

  def trace_handler(prof):
    alog.debug2(prof.key_averages().table(
      sort_by=sort_by, row_limit=row_limit))
    if traces_path:
      # Use chrome://tracing for viewing.
      prof.export_chrome_trace(os.path.join(traces_path, f'trace_{prof.step_num}.json'))
    if stacks_path:
      # git clone https://github.com/brendangregg/FlameGraph
      # cd FlameGraph
      # ./flamegraph.pl --title "CUDA time" --countname "us." stack_?.txt > perf_viz.svg
      prof.export_stacks(os.path.join(stacks_path, f'stack_{prof.step_num}.txt'))

  return trace_handler


def create_profiler(args):
  prof_args = pyu.get_fn_kwargs(args, torch.profiler.profile, prefix='prof')

  activities = [torch.profiler.ProfilerActivity.CPU]
  if torch.cuda.is_available():
    activities.append(torch.profiler.ProfilerActivity.CUDA)

  schedule = torch.profiler.schedule(
    skip_first=pyu.dget(args, 'prof.skip_first', 0),
    wait=pyu.dget(args, 'prof.wait', 16),
    warmup=pyu.dget(args, 'prof.warmup', 2),
    active=pyu.dget(args, 'prof.active', 2),
    repeat=pyu.dget(args, 'prof.repeat', 0))

  trace_handler = _make_trace_handler(args)

  prof_args.update(activities=activities, schedule=schedule,
                   on_trace_ready=trace_handler)

  return torch.profiler.profile(**prof_args)


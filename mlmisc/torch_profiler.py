import os

import py_misc_utils.alog as alog
import py_misc_utils.assert_checks as tas
import py_misc_utils.core_utils as pycu
import py_misc_utils.inspect_utils as pyiu
import torch


class NoopProfiler:

  def __enter__(self):
    return self

  def __exit__(self, *exc):
    return False

  def step(self):
    pass


def _make_trace_handler(args):
  # https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html
  # https://pytorch.org/docs/stable/profiler.html
  row_limit = pycu.dget(args, 'prof.row_limit', 10)
  sort_by = pycu.dget(args, 'prof.sort_by', None)
  if sort_by is None:
    sort_by = 'self_cuda_time_total' if torch.cuda.is_available() else 'cpu_time_total'
  traces_path = pycu.dget(args, 'prof.traces_path', None)
  stacks_path = pycu.dget(args, 'prof.stacks_path', None)

  def trace_handler(prof):
    alog.debug2(prof.key_averages().table(
      sort_by=sort_by, row_limit=row_limit))
    if traces_path is not None:
      # Use chrome://tracing for viewing.
      prof.export_chrome_trace(os.path.join(traces_path, f'trace_{prof.step_num}.json'))
    if stacks_path is not None:
      # git clone https://github.com/brendangregg/FlameGraph
      # cd FlameGraph
      # ./flamegraph.pl --title "CUDA time" --countname "us." stack_?.txt > perf_viz.svg
      prof.export_stacks(os.path.join(stacks_path, f'stack_{prof.step_num}.txt'))

  return trace_handler


def create_profiler(args):
  prof_args = pyiu.get_fn_kwargs(args, torch.profiler.profile, prefix='prof')

  activities = [torch.profiler.ProfilerActivity.CPU]
  if torch.cuda.is_available():
    activities.append(torch.profiler.ProfilerActivity.CUDA)

  schedule = torch.profiler.schedule(
    skip_first=pycu.dget(args, 'prof.skip_first', 0),
    wait=pycu.dget(args, 'prof.wait', 8),
    warmup=pycu.dget(args, 'prof.warmup', 2),
    active=pycu.dget(args, 'prof.active', 2),
    repeat=pycu.dget(args, 'prof.repeat', 0))

  trace_handler = _make_trace_handler(args)

  prof_args.update(activities=activities,
                   schedule=schedule,
                   on_trace_ready=trace_handler)

  return torch.profiler.profile(**prof_args)


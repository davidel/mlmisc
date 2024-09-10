import importlib
import os
import shutil
import subprocess
import sys
import tempfile
import urllib

import py_misc_utils.alog as alog
import py_misc_utils.utils as pyu
import torch
import torch.nn as nn


def _clone_repo(repo, root, force_clone):
  pr = urllib.parse.urlparse(repo)
  upath = pr.path[1: ] if pr.path.startswith('/') else pr.path

  rpath = os.path.join(root, upath)
  if force_clone and os.path.isdir(rpath):
    shutil.rmtree(rpath)
  if not os.path.isdir(rpath):
    alog.debug(f'Cloning repo: {repo}')
    subprocess.check_call(['git', 'clone', '-q', '--depth', '1', repo, rpath])

  return rpath


def _add_python_paths(path):
  with os.scandir(path) as sdir:
    for de in sdir:
      if de.is_dir():
        if os.path.isfile(os.path.join(de.path, '__init__.py')):
          pyu.maybe_add_path(sys.path, path)
        else:
          _add_python_paths(de.path)


def _load_module(rpath, module):
  mpath = os.path.join(rpath, module)
  if os.path.isfile(mpath):
    mod = pyu.load_module(mpath, add_sys_path=True)
  else:
    _add_python_paths(rpath)
    mod = importlib.import_module(module)

  return mod


class WebModule(nn.Module):

  def __init__(self, repo, module, ctor,
               root=None,
               force_clone=False,
               mod_args=None,
               mod_kwargs=None):
    root = root or os.getenv('MODULES_ROOT',
                             os.path.join(os.getenv('HOME', '.'), 'module_repos'))
    mod_args = mod_args or ()
    mod_kwargs = mod_kwargs or {}

    rpath = _clone_repo(repo, root, force_clone)

    mod = _load_module(rpath, module)

    super().__init__()
    self.net = getattr(mod, ctor)(*mod_args, **mod_kwargs)

  def forward(self, *args, **kwargs):
    return self.net(*args, **kwargs)


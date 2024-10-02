import importlib
import os
import urllib

import py_misc_utils.alog as alog
import py_misc_utils.assert_checks as tas
import py_misc_utils.git_repo as pygr
import py_misc_utils.module_utils as pymu
import py_misc_utils.utils as pyu
import torch
import torch.nn as nn


def _clone_repo(repo, root, force_clone, commit):
  pr = urllib.parse.urlparse(repo)
  upath = pr.path[1:] if pr.path.startswith('/') else pr.path

  rpath = os.path.join(root, upath)

  git = pygr.GitRepo(rpath)
  git.clone(repo, force=force_clone, shallow=commit is None)
  if commit is not None:
    git.checkout(commit)

  return rpath


def _add_python_paths(path):
  with os.scandir(path) as sdir:
    for de in sdir:
      if de.is_dir():
        if os.path.isfile(os.path.join(de.path, '__init__.py')):
          pymu.add_sys_path(path)
        else:
          _add_python_paths(de.path)


def _load_module(rpath, module):
  mpath = os.path.join(rpath, module)
  if os.path.isfile(mpath):
    mod = pymu.load_module(mpath, add_syspath=True)
  else:
    _add_python_paths(rpath)
    mod = importlib.import_module(module)

  return mod


def _modules_cachedir():
  cache_dir = os.getenv('MODULES_CACHEDIR', None)
  if cache_dir is None:
    cache_dir = os.path.join(os.getenv('HOME', '.'), 'module_repos')

  return cache_dir


class WebModule(nn.Module):

  def __init__(self, repo, module, ctor,
               cache_dir=None,
               commit=None,
               force_clone=None,
               mod_args=None,
               mod_kwargs=None):
    cache_dir = pyu.value_or(cache_dir, _modules_cachedir())
    force_clone = pyu.value_or(force_clone, False)
    mod_args = pyu.value_or(mod_args, ())
    mod_kwargs = pyu.value_or(mod_kwargs, {})

    alog.debug(f'Using Web Modules cache folder "{cache_dir}"')

    rpath = _clone_repo(repo, cache_dir, force_clone, commit)

    mod = _load_module(rpath, module)

    super().__init__()
    self.repr_args = dict(repo=repo,
                          module=module,
                          ctor=ctor,
                          mod_args=mod_args,
                          mod_kwargs=mod_kwargs)
    self.net = getattr(mod, ctor)(*mod_args, **mod_kwargs)

  def forward(self, *args, **kwargs):
    return self.net(*args, **kwargs)

  def extra_repr(self):
    return pyu.stri(self.repr_args)


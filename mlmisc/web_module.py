import importlib
import os
import shutil
import subprocess
import sys
import tempfile
import urllib

import py_misc_utils.alog as alog
import py_misc_utils.assert_checks as tas
import py_misc_utils.utils as pyu
import torch
import torch.nn as nn


class GitRepo:

  def __init__(self, path):
    self.path = path

  def _git(self, *cmd):
    git_cmd = ['git', '-C', self.path] + list(cmd)
    alog.debug(f'Running GIT: {git_cmd}')

    return git_cmd

  def _cmd(self, *cmd):
    subprocess.run(self._git(*cmd), capture_output=True, check=True)

  def _outcmd(self, *cmd):
    return subprocess.check_output(self._git(*cmd))

  def repo(self):
    return self._outcmd('config', '--get', 'remote.origin.url')

  def clone(self, repo, force=False, shallow=False):
    do_clone = True
    if os.path.isdir(self.path):
      tas.check_eq(repo, self.repo(), f'Repo mismatch!')
      if force or shallow != self.is_shallow():
        alog.info(f'Purging old GIT folder: {self.path}')
        shutil.rmtree(self.path)
        os.mkdir(self.path)
      else:
        self.pull()
        do_clone = False

    if do_clone:
      if shallow:
        self._cmd('clone', '-q', '--depth', '1', repo)
      else:
        self._cmd('clone', '-q', repo)

  def current_commit(self):
    return self._outcmd('rev-parse', 'HEAD')

  def is_shallow(self):
    return self._outcmd('rev-parse', '--is-shallow-repository') == 'true'

  def pull(self):
    self._cmd('pull', '-q')

  def checkout(self, commit):
    self._cmd('checkout', '-q', commit)



def _clone_repo(repo, root, force_clone, commit):
  pr = urllib.parse.urlparse(repo)
  upath = pr.path[1: ] if pr.path.startswith('/') else pr.path

  rpath = os.path.join(root, upath)

  git = GitRepo(rpath)
  git.clone(repo, force=force_clone, shallow=commit is None)
  if commit is not None:
    git.checkout(commit)

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
               commit=None,
               force_clone=False,
               mod_args=None,
               mod_kwargs=None):
    root = root or os.getenv('MODULES_ROOT',
                             os.path.join(os.getenv('HOME', '.'), 'module_repos'))
    mod_args = mod_args or ()
    mod_kwargs = mod_kwargs or {}

    rpath = _clone_repo(repo, root, force_clone, commit)

    mod = _load_module(rpath, module)

    super().__init__()
    self.net = getattr(mod, ctor)(*mod_args, **mod_kwargs)

  def forward(self, *args, **kwargs):
    return self.net(*args, **kwargs)


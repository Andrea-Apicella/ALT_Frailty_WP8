import glob
import os

from natsort import natsorted


def listdir_nohidden_sorted(path) -> list:
    """Returns a list of the elements in the specified path, sorted by name. Skips dotfiles."""
    l = glob.glob(os.path.join(path, '*'))
    if len(l) > 0:
        return natsorted(l)
    else:
        raise Exception(f'List is empty. Invalid path {path}')


def safe_mkdir(path) -> None:
    """If does not already exists, makes a directory in the specified path."""
    if not os.path.exists(path):
        os.makedirs(path)

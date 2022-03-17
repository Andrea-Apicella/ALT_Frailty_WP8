import os
from natsort import natsorted
import glob


def listdir_nohidden_sorted(path):
    '''Returns a list of the elements in the specified path, sorted by name. Skips dotfiles.'''
    return natsorted(glob.glob(os.path.join(path, '*')))


def safe_mkdir(path):
    '''If does not already exists, makes a directory in the specified path.'''
    try:
        os.mkdir(path)
    except FileExistsError:
        print('Folder already exists, skipping creation.')

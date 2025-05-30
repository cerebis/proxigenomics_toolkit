from distutils.sysconfig import get_python_lib
import numpy as np
import os
import sys


def make_random_seed():
    """
    Provide a random seed value between 1 and 10 million.
    :return: integer random seed
    """
    return np.random.randint(1000000, 10000000)


def make_dir(path, exist_ok=False):
    """
    Convenience method for making directories with a standard logic.
    An exception is raised when the specified path exists and is not a directory.
    :param path: target path to create
    :param exist_ok: if true, an existing directory is ok. Existing files will still cause an exception
    """
    if not os.path.exists(path):
        os.mkdir(path)
    elif not exist_ok:
        raise IOError('output directory already exists!')
    elif os.path.isfile(path):
        raise IOError('output path already exists and is a file!')


def app_path(subdir, filename):
    """
    Return path to named executable in a subdirectory of the running application

    :param subdir: subdirectory of application path
    :param filename: name of file
    :return: absolute path
    """
    return os.path.join(sys.path[0], subdir, filename)


def package_path(subdir, filename):
    """
    Return the path to a file within the installed location of the containing package.

    :param subdir: subdirectory of application path
    :param filename: name of file
    :return: absolute path
    """
    return os.path.join(get_python_lib(), __name__.split('.')[0], subdir, filename)


def exe_exists(exe_name):
    """
    Check that a executable exists on the Path.
    :param exe_name: the base executable name
    :return: True, an executable file named exe_name exists and has executable bit set
    """
    p, f = os.path.split(exe_name)
    assert not p, 'include only the base file name, no path specification'

    for pn in os.environ["PATH"].split(':'):
        full_path = os.path.join(pn, exe_name)
        if os.path.isfile(full_path) and os.access(full_path, os.X_OK):
            return True
    return False

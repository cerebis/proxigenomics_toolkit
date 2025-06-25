import os
import pytest
from proxigenomics_toolkit.misc_utils import make_random_seed, make_dir, exe_exists, package_path  # Updated import

def test_make_random_seed_range():
    seed = make_random_seed()
    assert 1000000 <= seed < 10000000, "Seed is not in the expected range of 1,000,000 to 10,000,000."

def test_make_random_seed_is_integer():
    seed = make_random_seed()
    assert isinstance(seed, int), "Seed is not an integer."


def test_exe_exists_with_existing_executable():
    assert exe_exists("ls") is True, "'ls' executable should exist on the system."

def test_exe_exists_with_nonexistent_executable():
    assert exe_exists("nonexistent_executable_12345") is False, "Non-existent executable returned True."

def test_exe_exists_with_path_in_executable():
    with pytest.raises(AssertionError, match="include only the base file name, no path specification"):
        exe_exists("/usr/bin/ls")


def test_make_dir_creates_directory(tmp_path):
    path = os.path.join(tmp_path, "test_dir")
    make_dir(path)
    assert os.path.isdir(path), "Directory was not created."


def test_make_dir_with_exist_ok(tmp_path):
    path = os.path.join(tmp_path, "test_dir")
    make_dir(path)
    make_dir(path, exist_ok=True)
    assert os.path.isdir(path), "Directory does not exist after calling make_dir with exist_ok=True."


def test_make_dir_raises_exception_for_existing_file(tmp_path):
    path = os.path.join(tmp_path, "test_file")
    with open(path, 'w') as f:
        f.write("test")
    with pytest.raises(OSError, match="output path already exists and is a file!"):
        make_dir(path)


def test_make_dir_raises_exception_for_existing_directory(tmp_path):
    path = os.path.join(tmp_path, "test_dir")
    make_dir(path)
    with pytest.raises(OSError, match="output directory already exists!"):
        make_dir(path)


def test_package_path_valid_path():
    result = package_path("numpy", "filename.txt")
    assert isinstance(result, str), "Result must be a string."
    assert result.endswith(os.path.join("numpy", "filename.txt")), "Path must include subdir and filename."

@pytest.mark.parametrize("subdir", ["", None])
def test_package_path_empty_subdir(subdir):
    with pytest.raises(AssertionError, match="subdir cannot be empty"):
        package_path(subdir, "filename.txt")

@pytest.mark.parametrize("filename", ["", None])
def test_package_path_empty_filename(filename):
    with pytest.raises(AssertionError, match="filename cannot be empty"):
        package_path("numpy", filename)

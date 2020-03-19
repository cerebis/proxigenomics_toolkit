from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext as build_ext_orig
from shutil import copyfile
import subprocess
import os
import re

pkg_name = 'proxigenomics_toolkit'

with open('README.md', 'r') as fh:
    long_description = fh.read()

version_str = None
VERSION_FILE = "{}/_version.py".format(pkg_name)
with open(VERSION_FILE, "rt") as vh:
    for _line in vh:
        mo = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", _line, re.M)
        if mo:
            version_str = mo.group(1)
            break

if version_str is None:
    raise RuntimeError("Unable to find version string in {}".format(VERSION_FILE))


class TarballExtension(Extension, object):
    """
    Simple class for use by build_tarball
    """
    def __init__(self, name, url, exe_path):
        """
        :param name: a name for the extension. This is used for naming the tarball and extracted parent path
        :param url: the remote location of the tarball (github)
        :param exe_path: a relative path to the executable from within the build directory
        """
        super(TarballExtension, self).__init__(name, sources=[])
        self.url = url
        self.exe_path = exe_path

        # attempt to use GNU tar and not Mac OSX bsd tar or the like
        self.tar_cmd = 'tar'
        if re.search(r'GNU', subprocess.check_output([self.tar_cmd, '--version'])) is None:
            self.tar_cmd = 'gtar'
        if re.search(r'GNU', subprocess.check_output([self.tar_cmd, '--version'])) is None:
            raise IOError('GNU tar was not found and installation requires special features')

    @property
    def tarball(self):
        """
        :return: a name for the tarball based on the extension name
        """
        return '{}_tarball.tar.gz'.format(self.name)

    @property
    def exe_name(self):
        """
        :return: the executable file name without path
        """
        return os.path.basename(self.exe_path)


class build_tarball(build_ext_orig, object):
    """
    Build a C/C++ Make projects from remote tarballs and place the binaries in proxigenomics_toolkit/external

    Build at install time allows easier support of runtime architectures which vary widely in age, making
    supplying a universal static binaries for external helpers difficult.
    """
    def run(self):
        for ext in self.extensions:
            self.build_tarball(ext)

    def curl_exists():
        try:
            subprocess.check_call(['curl', '--version'])
            return True
        except subprocess.CalledProcessError as e:
            return False
                                   
    def wget_exists():
        try:
            subprocess.check_call(['wget', '--version'])
            return True
        except subprocess.CalledProcessError as e:
            return False
                                   
    def build_tarball(self, ext):

        # fetch the relevant commit from github
        if curl_exists():
            self.spawn(['curl', '-L', ext.url, '-o', ext.tarball])
        elif wget_exists():
            self.spawn(['wget', '-O', ext.tarball, ext.url])
        else:
            raise IOError('Building {} requires either curl or wget be installed'.format(pkg_name))

        build_dir = os.path.join(self.build_lib, pkg_name)
        # rename parent folder to something simple and consistent
        self.spawn([ext.tar_cmd, '--transform=s,[^/]*,{},'.format(ext.name), '-xzvf', ext.tarball])
        # build
        self.spawn(['make', '-j4', '-C', ext.name])
        # copy the built binary to package folder
        src = os.path.join(ext.name, ext.exe_path)
        dest = os.path.join(build_dir, 'external', ext.exe_name)
        copyfile(src, dest)
        os.chmod(dest, 0o555)


setup(
    name=pkg_name,
    version=version_str,
    author='Matthew Z DeMaere',
    author_email='matt.demaere@gmail.com',
    description='Tools for 3C-based sequencing',
    long_description=long_description,
    url='https://github.com/cerebis/proxigenomics_toolkit',
    packages=find_packages(),
    license='GNU Affero General Public License v3',
    platforms='Linux-86_x64',
    include_package_data=True,
    zip_safe=False,

    ext_modules=[TarballExtension('infomap',
                                  'https://github.com/mapequation/infomap/tarball/11bd312db1',
                                  'Infomap'),
                 TarballExtension('lkh',
                                  'https://github.com/cerebis/LKH3/tarball/master',
                                  'LKH')],
    cmdclass={
        'build_ext': build_tarball
    },

    setup_requires=['numpy<1.15.0'],

    install_requires=['biopython',
                      'matplotlib<3',
                      'networkx<2',
                      'numba<=0.42.0',
                      'numpy<1.15.0',
                      'python-louvain',
                      'pysam',
                      'PyYAML<4',
                      'scipy',
                      'seaborn',
                      'sparse',
                      'tqdm',
                      'typing',
                      'llvmlite<0.27.1',
                      'cython',
                      'm3-cdecimal',
                      'lap @ git+https://github.com/gatagat/lap@master#egg=lap-99',
                      'polo @ git+https://github.com/adrianveres/polo@master#egg=polo-99'
                      ],

    dependency_links=['git+https://github.com/gatagat/lap@master#egg=lap-99',
                      'git+https://github.com/adrianveres/polo@master#egg=polo-99'
                      ],

    classifiers=[
        'Programming Language :: Python :: 2.7',
        'License :: OSI Approved :: GNU Affero General Public License v3',
        'Operating System :: POSIX :: Linux',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Bio-Informatics',
        'Development Status :: 2 - Pre-Alpha'
    ]

)

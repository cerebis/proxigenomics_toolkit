from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext as build_ext_orig
from shutil import copyfile
import os
import re

pkg_name = 'proxigenomics_toolkit'

# Infomap source details
infomap_url = 'https://github.com/mapequation/infomap/tarball/'
infomap_commit = '11bd312db1'
infomap_tarball = 'infomap.tar.gz'

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


class InfomapExtension(Extension, object):

    def __init__(self, name):
        super(InfomapExtension, self).__init__(name, sources=[])


class build_ext(build_ext_orig, object):

    def run(self):
        for ext in self.extensions:
            self.build_infomap(ext)

    def build_infomap(self, ext):
        build_dir = os.path.join(self.build_lib, pkg_name)
        # fetch the relevant commit from github
        self.spawn(['curl', '-L', '{}{}'.format(infomap_url, infomap_commit), '-o', infomap_tarball])
        # rename parent folder to something simple and consistent
        self.spawn(['tar', '--transform=s,[^/]*,infomap,', '-xzvf', infomap_tarball])
        # build infomap
        self.spawn(['make', '-j4', '-C', 'infomap'])
        # copy the infomap binary to package folder
        src = os.path.join('infomap', 'Infomap')
        dest = os.path.join(build_dir, 'external', 'Infomap')
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

    ext_modules=[InfomapExtension('{}/external/infomap'.format(pkg_name))],
    cmdclass={
        'build_ext': build_ext
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

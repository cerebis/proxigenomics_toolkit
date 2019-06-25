from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext as build_ext_orig
from shutil import copyfile
import os

__pkg_name__ = 'proxigenomics_toolkit'
__infomap_url__ = 'https://github.com/mapequation/infomap/tarball/'
__infomap_commit__ = '11bd312db1'
__infomap_tarball__ = 'infomap.tar.gz'


class InfomapExtension(Extension, object):

    def __init__(self, name):
        super(InfomapExtension, self).__init__(name, sources=[])


class build_ext(build_ext_orig, object):

    def run(self):
        for ext in self.extensions:
            self.build_infomap(ext)

    def build_infomap(self, ext):
        build_dir = os.path.join(self.build_lib, __pkg_name__)
        # fetch the relevant commit from github
        self.spawn(['curl', '-L', '{}{}'.format(__infomap_url__, __infomap_commit__), '-o', __infomap_tarball__])
        # rename parent folder to something simple and consistent
        self.spawn(['tar', '--transform=s,[^/]*,infomap,', '-xzvf', __infomap_tarball__])
        # build infomap
        self.spawn(['make', '-j4', '-C', 'infomap'])
        # copy the infomap binary to package folder
        src = os.path.join('infomap', 'Infomap')
        dest = os.path.join(build_dir, 'external', 'Infomap')
        copyfile(src, dest)
        os.chmod(dest, 0o555)


with open('README.md', 'r') as fh:
    long_description = fh.read()

setup(
    name=__pkg_name__,
    version='0.1a2',
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

    ext_modules=[InfomapExtension('{}/external/infomap'.format(__pkg_name__))],
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

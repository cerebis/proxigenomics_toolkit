import setuptools

with open('README.md', 'r') as fh:
    long_description = fh.read()


setuptools.setup(
    name='proxigenomics_toolkit',
    version='0.1a2',
    author='Matthew Z DeMaere',
    author_email='matt.demaere@gmail.com',
    description='Tools for 3C-based sequencing',
    long_description=long_description,
    url='https://github.com/cerebis/proxigenomics_toolkit',
    packages=setuptools.find_packages(),
    license='GNU Affero General Public License v3',
    platforms='Linux-86_x64',
    include_package_data=True,

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

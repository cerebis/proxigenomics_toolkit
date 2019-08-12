# proxigenomics_toolkit
A collection of classes and methods for the analysis of 3C-based sequencing data.

The toolkit forms the basis of the 3C-based tools [bin3C](https://github.com/cerebis/bin3C/tree/pgtk) and [scaffold3C](https://github.com/cerebis/scaffold3C).

### Basic requirements:
- Python 2.7
- Pip >=19

#### Dependency based requirements
A number of toolkit dependencies are built from source and as such bring along additional dependencies. It is possible that your chosen system may already have these packages installed. They can be fulfilled using your distribution package manager (yum, apt-get) or through userland managers like Conda.

- C/C++ compiler fully supporting C++ 11 (GCC >=4.8.1)
- Development packages for:
  - zlib
  - gzip
  - bzip2
  - curl 
  - openssl


## Installation

pgtk is not expected to be installed alone and, rather, is a simple shared API used in some of our Hi-C projects (bin3C, scaffold3C).

You can, however, install pgtk using recent versions (>19) of Pip as follows:

Within a Python 2.7 virtual environment

```bash
# first install NumPy<1.15 and Cython
bin/pip install "numpy<1.15" cython

# install pgtk
bin/pip install git+https://github.com/cerebis/proxigenomics_toolkit
```
### External tools

The toolkit makes use of external software tools which are compiled from source. This step introduces the above mention additional requirements for the successful installation of pgtk, while the benefit of building these tools from source is that we can support a wider range of runtime environments.

- Infomap
- LKH

The compiled tools are stored within the package hierarchy at proxigenomics_toolkit/external/

## Projects

### bin3C

bin3C extracts metgenome-assembled genomes (MAGs) from metagenomic datasets using Hi-C linkage information.

### scaffold3C

scaffold3C using the same Hi-C linkage information to order and potentially orientate assembly fragments (contigs or scaffolds). Scaffolding can be applied downstream of bin3C (to each sufficiently large MAG) or directly on the assembly of a mono-chromosomal genome.


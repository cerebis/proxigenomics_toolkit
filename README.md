# proxigenomics_toolkit
A collection of classes and methods for the analysis of 3C-based sequencing data.

The toolkit forms the basis of the 3C-based tools [bin3C](https://github.com/cerebis/bin3C/tree/pgtk) and [scaffold3C](https://github.com/cerebis/scaffold3C).

### Requirements:
- Python 2.7
- virtualenv
- Pip >=19
- C/C++ compiler. (tested with GNU C/C++)

## Installation

pgtk is not expected to be installed directly, but rather acts as a simply API within some of our Hi-C projects (bin3C, scaffol3C).

You can, however, install pgtk using recent versions (>19) of Pip as follows:

Within a Python 2.7 virtual environment

```bash
# first install NumPy<1.15 and Cython
bin/pip install "numpy<1.15" cython

# install pgtk
bin/pip install git+https://github.com/cerebis/proxigenomics_toolkit
```
### Binary Helpers

The toolkit makes use of a few external pre-compiled programs, which are currently supplied as staticaly linked binaries for recent Linux x86_64 kernels. Older systems may run into trouble if these tools are invoked. We wish to provide source builds of these tools but, before we do so, there are outstanding software license issues to consider.

- Infomap
- mcl
- LKH

Pre-compiled tools are stored within the package hierarchy at proxigenomics_toolkit/external/

## Projects

### bin3C

bin3C extracts metgenome-assembled genomes (MAGs) from metagenomic datasets using Hi-C linkage information.

### scaffold3C

scaffold3C using the same Hi-C linkage information to order and potentially orientate assembly fragments (contigs or scaffolds). Scaffolding can be applied downstream of bin3C (to each sufficiently large MAG) or directly on the assembly of a mono-chromosomal genome.


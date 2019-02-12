# proxigenomics_toolkit
A collection of classes and methods for the analysis of 3C-based sequencing data.

The toolkit forms the basis of the 3C-based tools [bin3C](https://github.com/cerebis/bin3C) and [scaffold3C](https://github.com/cerebis/scaffold3C).

##Installation
###Dependencies
The codebase possesses two python dependencies which cannot be installed from PyPi and instead are pulled from github.

- lap
- polo

##Binary Helpers
The toolkit uses external programs which are currently supplied in the form of pre-built binaries for Linux x86_64. Although these are statically linked, they may fail to run on older kernels.

- Infomap
- mcl
- LKH

These tools are stored within the proxgenomics_toolkit package hierarchy at proxigenomics_toolkit/external/


##bin3C
bin3C extracts metgenome-assembled genomes (MAGs) from metagenomic datasets using Hi-C linkage information.

##scaffold3C

scaffold3C using the same Hi-C linkage information to order and potentially orientate assembly fragments (contigs or scaffolds). Scaffolding can be applied downstream of bin3C (to each sufficiently large MAG) or directly on the assembly of a mono-chromosomal genome.


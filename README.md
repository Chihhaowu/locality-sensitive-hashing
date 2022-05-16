# LSH

## Description
**Question**: Given a large single-cell expression matrix E of dimensions M×N, where M is the number of cells and N is the number of reference genes. The goal of this project is to develop an efficient query scheme or index to allow for “expression pattern” queries, alternatively cell searching, taking vectors that represent gene expression counts as input. Read more: https://hackmd.io/@PI7Og0l1ReeBZu_pjQGUQQ/Hyhdz86J5


The script implements locality sensitive hashing using SimHash hash function. Returns the indices of E, representing the cell(s) which exhibit expression profiles that resemble the query vector of gene expression counts. To complement this approximate nearest neighbors (ANNs) search algorithm, there is a also a simple exact match function that does bitwise comparision all returned candidates to find matching full or partial expression patterns.

In the data folder are the script to generate the simulated RNA-seq expression data used in this project and a copy of the data itself from running the script.

## Requirements/Compatibility
Python 3.9.12

## Usage
Example code to execute all the evaluations/tests described are present ***thoroughly documented*** in the script.


## Reference
referenced https://github.com/dataplayer12/Fly-LSH to help implement SimHash/random projection LSH
referenced https://github.com/XuegongLab/DenseFly4scRNAseq/ to generate simulated scRNA-seq data

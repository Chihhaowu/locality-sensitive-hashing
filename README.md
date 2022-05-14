# LSH

## Description
**Reference**: an expression matrix E of dimensions m x n, where m are the reference cells and n are the gene expression level

**Query**: a logical conjunction in the form of (...)

Implements locality sensitive hashing using SimHash hash function. Returns the index(ces) of E, representing the cell(s) which exhibit expression profiles. To complement this approximate nearest neighbors (ANNs) search algorithm, there is a exact match method that compares bitwise all returned cells to find matching full or partial expression patterns.

## Requirements/Compatibility
Python 3.9.12

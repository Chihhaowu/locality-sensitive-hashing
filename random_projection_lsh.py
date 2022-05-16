import time
import math
import pandas as pd
import numpy as np
from functools import reduce
from sklearn.metrics import confusion_matrix, cohen_kappa_score

class RandomProjectionLSH():

    def __init__(self, data, hash_length, evaluation_labels=None):
        """
        data:  MxN expression matrix, M cells with N genes
        hash_length: size of the hash value
        evaluation_labels: labels of cell types (for evaluation)
        """
        # for functionality
        self.data_as_dataframe = data
        data = np.array(data)
        self.data = data - np.mean(data, axis=1, keepdims=True) # correction of gene expression counts, suggested in DenseFly and FlyHash
        numFeatures = data.shape[1]
        self.weights = SupportFunctions.dense_gaussian(numFeatures, hash_length)
        self.hashes = (self.data@self.weights)>=0
        self.bin_hashes()

        # for evaluation
        self.labels = evaluation_labels
        self.numLabels = np.unique(self.labels).shape[0]
    
    # assort cells/expression profiles based on their hash values into different bins; each bin contains all
    # similar expression vectors with the same hash value. Two dictionaries are also create to support multibin search.
    def bin_hashes(self): 
        self.distinct_hashes = np.unique(self.hashes, axis=0)
        numBins = self.distinct_hashes.shape[0]
        numRows = self.data.shape[0]

        hashBin = np.zeros(numRows, dtype=np.int16)
        for binNum, hash in enumerate(self.distinct_hashes):
            _bin = (self.hashes == hash).all(axis=1)
            for idx in np.flatnonzero(_bin):
                hashBin[idx] = binNum

        self.cells_assignedToBins = {cell: bin for cell, bin in enumerate(hashBin)}
        self.bins_whichCells = {bin: np.flatnonzero(hashBin == bin) for bin in range(numBins)}
  
    # given some new cell/expression vector, hash and obtain the encoded bit vector. If the hash table has already
    # been associated with cell(s) in the reference, add the new cell to the same bin. Otherwise, create a new bin
    # and place in it the new cell.
    def update_hash_table(self, new_pattern: np.ndarray):
    
        if new_pattern.size != self.weights.shape[0]:
            raise ValueError("Invalid length, did not add new pattern to table")

        corrected_count_vector = new_pattern - np.mean(new_pattern)
        bv_new_pattern = (corrected_count_vector@self.weights) >= 0
        new_cell = max(self.cells_assignedToBins.keys()) + 1
        
        # use bitwise xor to find if hash value has previously been generated
        i = 0
        same_bv = False
        while same_bv == False:
            if i == self.distinct_hashes.shape[0]:
                # if the hash value is new, then update the following
                np.append(self.distinct_hashes, bv_new_pattern)
                new_bin =  max(self.cells_assignedToBins.values()) + 1
                self.cells_assignedToBins[new_cell] = new_bin 
                self.bins_whichCells[new_bin] = np.array([new_cell])
                return

            same_bv = (self.distinct_hashes[i] ^ bv_new_pattern).sum() == 0
            i += 1

        # notice the index of the distinct hash corresponds to the bin number
        bin = i-1
        self.cells_assignedToBins[new_cell] = [bin]
        self.bins_whichCells[bin] = np.append(self.bins_whichCells[bin], new_cell)
     
    # perform query of a vector of gene expression counts, looking at multiple bins that contain cells with similar
    # expression profiles. Searching consists of obtaining the hash value of the query and finding the hash values (bins)
    # that differ by some (small) number of bits; these other bins potentially contain cells with expression profiles 
    # that match that of the query. Returns a reduced set of indices from which an exact pattern search can be performed.
    def query(self, select_index: int, search_tolerance=1):
        """
        multibin query in the form of searching using one of the rows as the query pattern
        lookup-based; queries are guaranteed to have at least one match
        """
        query_hash_bin = self.cells_assignedToBins[select_index] # based on lookup, what is the bin/corresponding hash
        query_hash = self.distinct_hashes[query_hash_bin] # compute (for querying non-reference cells) or lookup the hash value of a cell

        bitwise_difference = (query_hash[np.newaxis,:] ^ self.distinct_hashes).sum(axis=1) 
        restricted_bins = np.flatnonzero(bitwise_difference <= search_tolerance) 
        candidates_from_bins = [self.bins_whichCells[i] for i in restricted_bins]
        candidate_matches = reduce(np.union1d, candidates_from_bins)

        return candidate_matches

    # similar to query. However, now consider that instead of caring about the counts of all genes, that we can ignore
    # the expression of some genes. At a high-level, what we want to do is to identify cells exhibiting more general patterns 
    # of gene expression. To do this, we consider the following approach. Since the expression of some genes are no longer relevant,
    # it is possible that cells that are not similar (when graphed, are distantly located) could have a similar expression pattern for
    # the genes that consider as important to define similarity. Here, we take the query and create several other queries where, at the
    # indices, of "non-important" genes, we sample randomly from a normal distribution N(mean expression of the gene, stddev). By doing
    # this, we ask for nearest neighbors several subspaces to find candidate matches. Then, similar to query, we search for exact matches.
    # Here, we also change the stringency of the query by increasing the search_tolerance, we accept more bins and thereby more profiles. 
    def query_extended(self, query_pattern: np.ndarray, feature_indices: list, search_tolerance=15):
        """
        Not yet optimised, but outlines approach.
        query_pattern: a vector of gene expression counts
        feature_indices: list of indices for the genes that are important for defining the similarity search
        search_tolerance: how many bits are allowed to differ between hash values
        """ 

        num_features = query_pattern.shape[0]
        substitute_indices = list(set(range(num_features)) - set(feature_indices))
    
        reference_means = np.mean(self.data, axis=0)
        reference_stddev = np.std(self.data, axis=0)

        all_candidates = set()
        for _ in range(100):
            for idx in substitute_indices:
                rand = np.random.default_rng().normal(reference_means[idx], reference_stddev[idx])
                query_pattern[idx] = math.floor(rand)

            query_hash = (query_pattern@self.weights) >= 0 # compute or lookup the hash value of a cell
            
            bitwise_difference = (query_hash[np.newaxis,:] ^ self.distinct_hashes).sum(axis=1)# bitwise, how much does the query hash value differ from other bins
            restricted_bins = np.flatnonzero(bitwise_difference <= search_tolerance)# return the indices of the bins which difference is within defined tolerances
            candidates_from_bins = [self.bins_whichCells[i] for i in restricted_bins]
            candidate_matches = reduce(np.union1d, candidates_from_bins)
            [all_candidates.add(match) for match in candidate_matches]
 
        return all_candidates
            

    def exact_matching(self, select_index: int, candidate_indices: np.ndarray, feature_indices: list):
        """
        looks in limited bins for exact pattern matches at specified positions, corresponding
        to different features
        """
        exact_matches = []
        compared_features = feature_indices if feature_indices else list(range(self.data_as_dataframe.shape[1]))
        pattern = self.data_as_dataframe.iloc[select_index,compared_features]

        for idx in candidate_indices:
            if (pattern ^ self.data_as_dataframe.iloc[idx,compared_features]).sum() == 0:
                exact_matches.append(idx)

        return exact_matches

    # the following three evaluation methods are modified from 
    # perform query operation on each entry and return the confusion matrix
    def compute_confusion_matrix(self, indices):
        cm = np.zeros((self.numLabels,self.numLabels))
        for idx in indices:
            query_assigned_label = self.labels[self.query(idx)]
            true_label = np.array([self.labels[idx]]*query_assigned_label.shape[0])
            cm += confusion_matrix(true_label,query_assigned_label,labels=np.unique(self.labels))
        return cm

    def cohens_kappa_score(self, matrix):
        po = matrix.trace()/np.sum(matrix)
        pe = sum(np.sum(matrix, axis=0)*np.sum(matrix, axis=1))/np.sum(matrix)/np.sum(matrix)
        return (po-pe)/(1-pe)

    def evaluate_lsh(self):
        CKS = 0
        for _ in range (5):
            query_indices = np.random.choice(self.data.shape[0], self.data.shape[0]//5)
            CM = self.compute_confusion_matrix(query_indices)
            CKS += self.cohens_kappa_score(CM)/5
        return CKS

class SupportFunctions:

    @staticmethod
    def sample_random_normal(dims: tuple)->np.ndarray:
        """
        Return a one-dimensional array with numVal elements
        sampled from a gaussian distribution
        """
        rng =  np.random.default_rng()
        vals = rng.standard_normal(dims)
        return vals

    @staticmethod
    def dense_gaussian(numFeatures, hash_length):
        """
        For the length of the hash value, get the orthogonal
        vectors (normals) that when we multiply, obtain which
        side of the random hyperplanes the point lies in 
        """
        normal = SupportFunctions.sample_random_normal((numFeatures, hash_length))
        return normal

if __name__ == "__main__":
    
    # load sample RNA sequencing expression dataset; takes somewhat long to load each time
    data = pd.read_csv("/Users/dch/Downloads/SelfMapping.txt", delim_whitespace=True)
    df = data.iloc[:,:-1]
    labels = data.iloc[:, -1]

    """
    # sub pattern search
    test = RandomProjectionLSH(df, hash_length=128)
    q_pattern = df.iloc[0,:]
    # indices of features of query that we want exact matches for in the search
    feature_indices = np.random.default_rng().permutation(10000)[:8000]
    candidate_indices = test.query_extended(q_pattern, feature_indices)
    
    # add new cell/expression profile to reference
    test = RandomProjectionLSH(df, hash_length=8)
    existingPattern = np.zeros(10000) # replace with any vector representing the expression profile of a new cell
    test.update_hash_table(existingPattern)
    
    #single query; choose one cell from the reference as the query
    lsh_data = RandomProjectionLSH(data=df, hash_length=64)
    candidates = lsh_data.query(3) # find the bins which potential candidates lie in
    exact_matches = lsh_data.exact_matching(select_index=3, candidate_indices=candidates, feature_indices=[]) # use exact matching; bitwise comparison of patterns
    
    # evaluate LSH over different hash_lengths
    for hash_length in [16,32,64,128,256,512,1024]: 
        simhash = RandomProjectionLSH(data=df, hash_length=hash_length, evaluation_labels=labels)
        print(f"CKS using hash length {hash_length} is: {simhash.evaluate_lsh()}")
    
    # evaluate LSH query time over different hash lengths (optional exact matching; just linear scan)
    for hash_length in [32,64,128,512,1024]: 
        simhash = RandomProjectionLSH(data=df, hash_length=hash_length)

        start = time.time()
        for i in range(100):
            candidates = simhash.query(i)
            #simhash.exact_matching(select_index=i, candidate_indices=candidates, feature_indices=[])
        print(f"Time to query 100 patterns from 2000 cells without exact match from using model with hash length of {hash_length}: {time.time()-start}")
    
    # evaluate LSH query time over different number of cells
    numCells = 2000
    subset = np.random.choice(range(2000), numCells)
    df_subset = df.iloc[subset,:]
    labels_subset = labels.iloc[subset]

    hash_length = 128
    simhash = RandomProjectionLSH(data=df_subset, hash_length=hash_length)

    start = time.time()
    for i in range(100):
        simhash.query(i)
    print(f"Time to query 100 patterns from reference of {numCells} cells: {time.time()-start}")
   
    ##### evaluate time to construct LSH over different number of cells
    numCells = 500
    subset = np.random.choice(range(2000), numCells)
    df_subset = df.iloc[subset,:]
    labels_subset = labels.iloc[subset]
    hash_length = 128
    start = time.time()
    simhash = RandomProjectionLSH(data=df_subset, hash_length=hash_length)
    print(f"Time to construct the LSH model from reference of {numCells} cells: {time.time()-start}")
    """

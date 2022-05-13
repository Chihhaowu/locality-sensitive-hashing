import time
import pandas as pd
import numpy as np
from functools import reduce
from sklearn.metrics import confusion_matrix, cohen_kappa_score

class RandomProjectionLSH():

    def __init__(self, data, hash_length, nnn, evaluation_labels=None):
        
        # for functionality
        self.data_as_dataframe = data
        data = np.array(data)
        self.data = data - np.mean(data, axis=1, keepdims=True) # with correction
        numFeatures = data.shape[1]
        self.weights = SupportFunctions.dense_gaussian(numFeatures, hash_length)
        self.hashes = (self.data@self.weights)>0 # obtained by the dot product of each point to the normal of each hyperplane
        self.nnn = nnn
        self.bin_hashes()
        
        # for evaluation
        self.labels = evaluation_labels
        self.numLabels = np.unique(self.labels).shape[0]

    def bin_hashes(self): 
        """
        Create the bins for each hash value, then assort the
        queries into each of the bins
        """
        self.distinct_hashes = np.unique(self.hashes, axis=0) # unique hash values
        numBins = self.distinct_hashes.shape[0]
        numRows = self.data.shape[0]

        hashBin = np.zeros(numRows, dtype=np.int16) # np.zeros is more memory efficient than list; is data.shape same as unique_hashes.shape?
        for binNum, hash in enumerate(self.distinct_hashes): # assort each row into bin according to hash value
            _bin = (self.hashes == hash).all(axis=1)
            for idx in np.flatnonzero(_bin):
                hashBin[idx] = binNum

        self.cells_assignedToBins = {cell: bin for cell, bin in enumerate(hashBin)} # which bin has a cell been assigned to
        self.bins_whichCells = {bin: np.flatnonzero(hashBin == bin) for bin in range(numBins)} # which cells are present in each bin
        
    def query(self, select_index: int, search_tolerance=1):
        """
        Multiprobe query in the form of searching using one of the rows as the query pattern
        lookup-based; queries are guaranteed to have at least one match
        """
        query_hash_bin = self.cells_assignedToBins[select_index] # based on lookup, what is the bin/corresponding hash
        query_hash = self.distinct_hashes[query_hash_bin] # compute or lookup the hash value of a cell

        bitwise_difference = (query_hash[np.newaxis,:] ^ self.distinct_hashes).sum(axis=1)# bitwise, how much does the query hash value differ from other bins
        restricted_bins = np.flatnonzero(bitwise_difference <= search_tolerance)# return the indices of the bins which difference is within defined tolerances
        
        candidates_from_bins = [self.bins_whichCells[i] for i in restricted_bins]
        candidate_matches = reduce(np.union1d, candidates_from_bins)# fix this if results are unexpected
        
        #candidate_matches = candidate_matches[1:self.nnn+1] # legacy; REMOVE

        return candidate_matches

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
            #print(query_assigned_label, true_label) 
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
    
    # load sample RNA sequencing expression dataset
    data = pd.read_csv("/Users/dch/Downloads/SelfMapping.txt", delim_whitespace=True)
    df = data.iloc[:,:-1]
    labels = data.iloc[:, -1]

    # single query
    lsh_data = RandomProjectionLSH(data=df, hash_length=64, nnn=1)
    candidates = lsh_data.query(3) # find the bins which potential candidates lie in
    exact_matches = lsh_data.exact_matching(select_index=3, candidate_indices=candidates, feature_indices=[]) # use exact matching; bitwise comparison of patterns
    
    # evaluate LSH over different hash_lengths
    for hash_length in [16,32,64,128,256,512,1024]: 
        simhash = RandomProjectionLSH(data=df, hash_length=hash_length, nnn=1, evaluation_labels=labels)
        print(f"CKS using hash length {hash_length} is: {simhash.evaluate_lsh()}")
   
    # evaluate LSH query time over different hash lengths
    for hash_length in [32,64,128,512,1024]: 
        simhash = RandomProjectionLSH(data=df, hash_length=hash_length, nnn=1)

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
    simhash = RandomProjectionLSH(data=df_subset, hash_length=hash_length, nnn=1)

    start = time.time()
    for i in range(100):
        simhash.query(i)
    print(f"Time to query 100 patterns from reference of {numCells} cells: {time.time()-start}")
   
    
    # evaluate time to construct LSH over different number of cells
    numCells = 500
    subset = np.random.choice(range(2000), numCells)
    df_subset = df.iloc[subset,:]
    labels_subset = labels.iloc[subset]

    hash_length = 128
    start = time.time()
    simhash = RandomProjectionLSH(data=df_subset, hash_length=hash_length, nnn=1)
    print(f"Time to construct the LSH model from reference of {numCells} cells: {time.time()-start}")
    

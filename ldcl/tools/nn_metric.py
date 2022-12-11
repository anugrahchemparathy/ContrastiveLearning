from sklearn.neighbors import NearestNeighbors
from math import pi, ceil
import numpy as np

class Orbit_Evaluator():
    """
    Uses nearest neighbors on embedding space to predict conserved quantities to evaluate the quality of embeddings
    IMPORTANT: The variances and max deviations produced are not normalized because it requires the range of values
    """
    def __init__(self, encodings, conserved_quantities, splits = 10, shuffle = True):
        """
        expects encodings as (n,d) np array
        expects conserved quantities as list of length 3 of (n,) np arrays conserved quantities to test for
        expects in order phi_0, energy, angular momentum
        """
        self.encodings = encodings
        self.size = encodings.shape[0]
        for i,a in enumerate(conserved_quantities):
            assert a.shape[0] == self.size, f'{i}th element in conserved quantities does not match number of points'
        self.conserved_quantities = conserved_quantities
        assert splits < self.size, f'number of splits {splits} exceeds number of data points {self.size}'
        self.splits = splits
        if shuffle: # idk how to seed the random thing, will add it later
            p = np.random.permutation(self.size)
            self.encodings = self.encodings[p]
            for i, a in enumerate(self.conserved_quantities):
                self.conserved_quantities[i] = a[p]
        self.evaluated = False
        self.variance = None
        self.max_dev = None
    
    def evaluate(self):
        if self.evaluated:
            return
        
        self.variance = np.zeros(3)
        self.max_dev = np.zeros(3)
        
        batch_size = ceil(self.size / self.splits)
        
        for i in range(self.splits):
            # data split
            lower = i * batch_size
            upper = min(self.size, (i+1) * batch_size)
            train = np.concatenate((self.encodings[:lower], self.encodings[upper:]))
            test = self.encodings[lower:upper]
            vals = [(np.concatenate((a[:lower], a[upper:])), a[lower:upper]) for a in self.conserved_quantities] # pairs for training and test conserved quantities
            
            # nearest neighbors
            neigh = NearestNeighbors(n_neighbors = 1)
            neigh.fit(train) # fit to training values
            
            predicted = neigh.kneighbors(test, n_neighbors = 1, return_distance = False).flatten() # gets the index of the nearest neighbor to each of the test points as a (len(test),) np array
            
            for i, (X,Y) in enumerate(vals):
                diff = np.abs(X[predicted] - Y)
                if i == 0:
                    diff = np.array([min(x, 2*pi - x) for x in diff])
                self.max_dev[i] = max(self.max_dev[i], np.amax(diff))
                self.variance[i] += np.sum(np.square(diff))
        
        self.variance /= self.size
        
        self.evaluated = True
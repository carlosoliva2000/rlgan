# multilayer perceptron of any architecture

from typing import Optional
import numpy as np
# import matplotlib.pyplot as plt

class MLP:
    '''
    Class to define Multilayer Perceptrons.
    Declare instance with MLP(layers).
    '''
    def __init__(self, layers: list[int], chromosome: Optional[np.ndarray]=None):
        '''
        layers: a tuple with (ninputs, nhidden1, nhidden2, ... noutput)
        chromosome: a flatten list with all weights and biases to initialize the MLP
        '''
        self.layers = layers
        self.trace = False
        self.threshold = 5.0
        self.labels = None # text for the labels [input-list, output-list]
        self.W = [] # list of numpy matrices
        self.b = [] # list of numpy vectors
        self.size = 0
        self.lRMS = [] # hold all traced RMSs to draw graph
        self.laccuracy = [] # hold all traced accuracies to draw graph
        self.mean_fitness = -np.inf

        if chromosome is None:  
            for i in range(len(layers)-1):
                w = np.random.rand(layers[i],layers[i+1])-0.5
                b = np.random.rand(layers[i+1])-0.5
                self.W.append(w)
                self.b.append(b)
                self.size += layers[i] * layers[i+1] + layers[i+1] 
        else:
            for i in range(len(layers)-1):
                self.size += layers[i] * layers[i+1] + layers[i+1]
            self.from_chromosome(chromosome)

    def reset_fitness(self):
        self.mean_fitness = -np.inf
        
    def sigm (self, neta):
        return 1.0 / (1.0 + np.exp(-neta))
    
    def forward (self, x): # fast forward (optimized in time, but not use to train!)
        for i in range(len(self.b)):
            net = np.dot(x,self.W[i]) + self.b[i]
            x = self.sigm(net)
        return x
            
    def to_chromosome (self) -> np.ndarray:
        '''
        Convert weights and biases to a flatten list to use in AG.
        '''
        # ch = np.array([])
        ch = np.array([])
        for w,b in zip(self.W, self.b):
            # print(w.flatten())
            # add to ch the flatten weights and biases
            ch = np.append(ch, w.flatten())
            ch = np.append(ch, b.flatten())
            # ch.extend(w.flatten())  # ch += w.flatten().tolist()
            # ch.extend(b.flatten())  # ch += b.flatten()  #.tolist()
        return ch
    
    def from_chromosome (self, ch):
        '''
        Convert a flatten list (chromosome from a GA) to internal weights and biases.
        '''
        if len(ch) != self.size:
            print(self.size)
            raise ValueError("Chromosome legnth doesn't match architecture")
        self.W = []
        self.b = []
        pos = 0
        for i in range(len(self.layers)-1): # for each layer
            to = self.layers[i]*self.layers[i+1] # number of weight
            w = np.array(ch[pos:pos+to]).reshape(self.layers[i],self.layers[i+1])
            pos += to
            to = self.layers[i+1] # number of bias
            b = np.array(ch[pos:pos+to]).reshape(self.layers[i+1])
            pos += to
            
            self.W.append(w)
            self.b.append(b)
        return self

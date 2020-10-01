import wisardpkg as wp
import random
import numpy as np
import time
from astropy.stats import bootstrap
from astropy.utils import NumpyRNGContext

LOW_N = 5
HIGH_N = 31
MIN_SCORE = 0.1
GROW_INTERVAL = 100
MAX_DISCRIMINATOR_LIMIT = 10

class Bagging(object):
    
    def __init__(self, train_dataset, learners, partitions = 0.75, models = "heterogeneous"):
        self.train_dataset = train_dataset
        self.learners = learners
        self.nets = []
        self.partitions = int(len(train_dataset)*partitions)
        if(self.partitions == 0):
            self.partitions = 1
        self.models = models
        self.entry_size = len(train_dataset.get(0))
        self.training_time = 0
        self.ensemble()
        
    def random_model(self):
        if (random.randint(0,1))%2==0:
            net = wp.Wisard(np.random.randint(LOW_N, HIGH_N))
        else:
            discriminator_limit = np.random.randint(2, MAX_DISCRIMINATOR_LIMIT)
            net = wp.ClusWisard(np.random.randint(LOW_N, HIGH_N), MIN_SCORE, GROW_INTERVAL, discriminator_limit)
        return net
        
    def random_wisard(self):
        return wp.Wisard(np.random.randint(LOW_N, HIGH_N))
        
    def random_clus(self):
        discriminator_limit = np.random.randint(2, MAX_DISCRIMINATOR_LIMIT)
        return wp.ClusWisard(np.random.randint(LOW_N, HIGH_N), MIN_SCORE, GROW_INTERVAL, discriminator_limit)
    
    def generate_dataset(self):
        boot = []
        for i in range(len(self.train_dataset)):
            boot.append(i)
        with NumpyRNGContext(1):
            bootresult = bootstrap(np.array(boot), self.learners, self.partitions)

        dataset = []
        for samples in bootresult:
            d = wp.DataSet()
            for sample in samples:
                d.add(self.train_dataset.get(int(sample)), self.train_dataset.getLabel(int(sample)))
            dataset.append(d)

        return dataset
         
    def ensemble(self):
        dataset = self.generate_dataset()

        for i in range(0, self.learners):
            if(self.models == "heteronegous"):
                net = self.random_model()
            else:
                if(self.models == "wisard"):
                    net = self.random_wisard()
                else:
                    net = self.random_clus()
            training_time = time.time()
            net.train(dataset[i])
            self.training_time = self.training_time + time.time() - training_time
            self.nets.append(net)

    def get_training_time(self):
        return self.training_time

    def classify(self, test_dataset):
        results = []
        for i in range(0, len(self.nets)):
            r = self.nets[i].classify( test_dataset )
            results.append(r)
        results = np.array(results)
        fr = []
        for i in range( results.shape[1] ):
            un, c = np.unique( results[:,i], return_counts = True )
            fr.append( un[ np.argmax(c) ] )
        return fr

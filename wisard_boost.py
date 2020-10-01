import wisardpkg as wp
import random
import numpy as np
import time
from sklearn.metrics import accuracy_score

LOW_N = 2
HIGH_N = 31
MIN_SCORE = 0.1
GROW_INTERVAL = 100
MAX_DISCRIMINATOR_LIMIT = 10

class Boost(object):
    
    def __init__(self, train_dataset, validation_dataset, learners, models = "heterogeneous"):
        self.train_dataset = train_dataset
        self.validation_dataset = validation_dataset
        self.learners = learners
        self.nets = []
        self.entry_size = len(train_dataset.get(0))
        self.data_positions = list(range(0, len(self.train_dataset)))
        self.ensemble_weights = [0] * learners
        self.total_sum = 0
        self.models = models
        
        self.y_test = []
        for i in range(len(self.validation_dataset)):
            self.y_test.append(self.validation_dataset.getLabel(i))

        self.training_time = 0
        self.validation_time = 0
        
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
        local_data_positions = random.sample(self.data_positions, int(len(self.train_dataset)/self.learners))
        dataset = wp.DataSet()
        for i in range(0, len(local_data_positions)):
            self.data_positions.remove(local_data_positions[i])
            dataset.add(self.train_dataset.get(local_data_positions[i]), self.train_dataset.getLabel(local_data_positions[i]))
        return dataset    

    def validate(self, net):
        results = net.classify(self.validation_dataset)
        return accuracy_score(self.y_test, results)
        
    def normalize_weights(self):
        for i in range(0, len(self.ensemble_weights)):
            if(np.isnan(self.ensemble_weights[i])):
                self.ensemble_weights[i] = 0
        
        self.total_sum = sum(self.ensemble_weights)
        
        for i in range(0, len(self.ensemble_weights)):
            if(self.total_sum == 0):
                self.ensemble_weights[i] = 0
            else:
                self.ensemble_weights[i] = (self.ensemble_weights[i] * 100)/self.total_sum
         
    def ensemble(self):
        for i in range(0, self.learners):
            if(self.models == "rew"):
                net = self.random_wisard()
            else:
                if(self.models == "crew"):
                    net = self.random_clus()
                else:
                    net = self.random_model()
            training_time = time.time()
            net.train(self.generate_dataset())
            self.training_time = self.training_time + time.time() - training_time
            validation_time = time.time()
            self.ensemble_weights[i] = self.validate(net)
            self.validation_time = self.validation_time + time.time() - validation_time
            self.nets.append(net)
        self.normalize_weights
            
    def get_training_time(self):
        return self.training_time
    
    def get_validation_time(self):
        return self.validation_time

    def classify(self, test_dataset):
        results = []
        for i in range(0, len(test_dataset)):
            result = {}
            for j in range(0, len(self.nets)):
                test = wp.DataSet()
                bi = wp.BinInput(test_dataset.get(i))
                test.add(bi, test_dataset.getLabel(i))
                r = self.nets[j].classify(test)
                                
                if(r[0] in result):
                    result[r[0]] += self.ensemble_weights[j]
                else:
                    result[r[0]] = 0
                    
            results.append(max(result, key = result.get))
               
        return results

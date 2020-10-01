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

class BordaBagging(object):
    
    def __init__(self, train_dataset, learners, partitions = "undefined", voting = "borda0"):
        self.train_dataset = train_dataset
        self.learners = learners
        self.nets = []
        self.partitions = partitions
        if(partitions == "undefined"):
            self.partitions = int(len(train_dataset)/75)
        if(self.partitions == 0):
            self.partitions = 1
        self.entry_size = len(train_dataset.get(0))
        self.voting = voting
        self.training_time = 0
        self.ensemble()
        
    def random_wisard(self):
        return wp.ClusWisard(np.random.randint(LOW_N, HIGH_N), 0.1, 10, 1)
           
    def generate_dataset(self):
        boot = []
        for i in range(len(self.train_dataset)):
            boot.append(i)
        with NumpyRNGContext(1):
            bootresult = bootstrap(np.array(boot), self.learners, int(len(self.train_dataset)*self.partitions))

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
            net = self.random_wisard()
            training_time = time.time()
            net.train(dataset[i])
            self.training_time = self.training_time + time.time() - training_time
            self.nets.append(net)
    
    def get_training_time(self):
        return self.training_time

    @staticmethod
    def get_labels(out):
        labels = []
        for label in out[0]:
            labels.append(label)
        return labels
    
    @staticmethod
    def borda_count_0(scores, labels):
        score_labels = [0] * len(labels)
        for i in range(len(scores)):
            for j in range(len(labels)):
                if(scores[i] == labels[j]):
                    score_labels[j] += 1
    
        scores_template = sorted(set(score_labels))
        new_scores = []
        for i in range(len(score_labels)):
            vote = scores_template.index(score_labels[i])
            new_scores.append(vote/(len(labels)-1))
        return labels[new_scores.index(max(new_scores))]

    @staticmethod
    def borda_count_1(scores, labels):
        score_labels = [0] * len(labels)
        for i in range(len(scores)):
            for j in range(len(labels)):
                if(scores[i] == labels[j]):
                    score_labels[j] += 1
        
        scores_template = sorted(set(score_labels))
        new_scores = []
        for i in range(len(score_labels)):
            vote = scores_template.index(score_labels[i])
            new_scores.append((vote+1)/len(labels))
        return labels[new_scores.index(max(new_scores))]

    @staticmethod
    def dowdall(scores, labels):
        score_labels = [0] * len(labels)
        for i in range(len(scores)):
            for j in range(len(labels)):
                if(scores[i] == labels[j]):
                    score_labels[j] += 1
        
        scores_template = sorted(set(score_labels), reverse = True)
        new_scores = []
        for i in range(len(score_labels)):
            vote = scores_template.index(score_labels[i])
            new_scores.append(1/(vote+1))
        return labels[new_scores.index(max(new_scores))]

    def classify(self, test_dataset):
        results = []
        for i in range(0, len(test_dataset)):
            scores = []
            test = wp.DataSet()
            bi = wp.BinInput(test_dataset.get(i))
            test.add(bi, test_dataset.getLabel(i))
            for j in range(0, len(self.nets)):
                scores.append(self.nets[j].classify(test)[0])
            
            out = self.nets[0].getAllScores(test)
            labels = self.get_labels(out)
            
            result = 0
            if(self.voting == "borda0"):
                result = self.borda_count_0(scores, labels)
            else:
                if(self.voting == "borda1"):
                    result = self.borda_count_1(scores, labels)
                else:
                    result = self.dowdall(scores, labels)
            results.append(result)
               
        return results

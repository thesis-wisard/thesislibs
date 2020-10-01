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

class VotingBagging(object):
    
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
    def get_scores(out):
        label_dic = []
        votes = []

        for label in out[0]:
            label_dic.append(label)
            votes.append(out[0][label][0])

        #print(label_dic)
        #print(votes)

        ordered_scores = sorted(set(votes), reverse = True)

        new_votes = []
        for i in range(len(votes)):
            vote = ordered_scores.index(votes[i])
            new_votes.append(vote)

        #print(new_votes)

        scores = []
        for i in range(len(new_votes)):
            if(i in new_votes):
                scores.append(label_dic[new_votes.index(i)])
            else:
                scores.append(label_dic[new_votes.index(max(new_votes))])

        return scores

    @staticmethod
    def plurality1(scores, labels):
        #print(scores)
        #print(labels)
        for i in range(len(labels)):
            best_candidate = 0
            best_score = 0
            for j in range(len(labels)):
                aux = scores[j].count(i)
                tie = False
                if(aux > best_score):
                    best_candidate = j
                    best_score = aux
                    if(tie):
                        tie = False
                elif(aux == best_score):
                    tie = True
            if(not(tie)):
                return labels[best_candidate]

    @staticmethod
    def plurality2(scores, labels):
        for i in range(len(labels)):
            best_candidate = 0
            second_candidate = 0
            best_score = 0
            #tiers = []
            for j in range(len(scores)):
                #print(scores)
                aux = scores[j].count(i)
                tie = False
                if(aux > best_score):
                    second_candidate = best_candidate
                    best_candidate = j
                    best_score = aux
                    tiers = [j]
                    if(tie):
                        tie = False
                elif(aux == best_score):
                    tie = True
                    #tiers.append(j)
            if(not(tie)):
                return labels[best_candidate]
            #scores = tiers
            new_scores = []
            for k in range(len(labels)):
                if(k == best_candidate):
                    new_scores.append(scores[best_candidate])
                else:
                    if(k == second_candidate):
                        new_scores.append(scores[second_candidate])
                    else:
                        new_scores.append([len(labels)] * len(scores[0]))
            scores = new_scores

    @staticmethod
    def plurality3(scores, labels):
        new_scores = []
        for i in range(len(scores)):
            new_scores.append(sum(scores[i]))
        return labels[new_scores.index(min(new_scores))]

    @staticmethod
    def plurality4(scores, labels, threshold):
        for i in range(len(labels)):
            best_candidate = 0
            second_candidate = 0
            best_score = 0
            tiers = []
            for j in range(len(scores)):
                #print(scores[j])
                aux = scores[j].count(i)
                tie = False
                if(aux > best_score):
                    second_candidate = best_candidate
                    best_candidate = j
                    best_score = aux
                    tiers = [j]
                    if(tie):
                        tie = False
                elif(aux == best_score):
                    tie = True
                    tiers.append(j)
            #print(best_score)
            if(not(tie) and (best_score > threshold * len(scores[0]))):
                return labels[best_candidate]
            new_scores = []
            for k in range(len(labels)):
                if(k == best_candidate):
                    new_scores.append(scores[best_candidate])
                else:
                    if(k == second_candidate):
                        new_scores.append(scores[second_candidate])
                    else:
                        new_scores.append([len(labels)] * len(scores[0]))
            scores = new_scores

    def classify(self, test_dataset):
        results = []
        for i in range(0, len(test_dataset)):
            votes = []
            scores = []
            test = wp.DataSet()
            bi = wp.BinInput(test_dataset.get(i))
            test.add(bi, test_dataset.getLabel(i))
            for j in range(0, len(self.nets)):
                votes.append(self.get_scores(self.nets[j].getAllScores(test)))
            
            labels = votes[0]

            for i in range(len(labels)):
                score = []
                for j in range(len(votes)):
                    if(labels[i] in votes[j]):
                        score.append(votes[j].index(labels[i]))
                    else:
                        score.append(votes[j].index(max(votes[j])))
                scores.append(score)
            
            result = 0
            if(self.voting == "plurality1"):
                result = self.plurality1(scores, labels)
            else:
                if(self.voting == "plurality2"):
                    result = self.plurality2(scores, labels)
                else:
                    if(self.voting == "plurality3"):
                        result = self.plurality3(scores, labels)
                    else:
                        result = self.plurality4(scores, labels, 0.3)
            results.append(result)
               
        return results

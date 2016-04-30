#!/usr/bin/python3
__author__ = 'kaiolae'
__author__ = 'kaiolae'
import Backprop_skeleton as Bp

import itertools, operator, random, statistics

#Class for holding your data - one object for each line in the dataset
class dataInstance:

    def __init__(self,qid,rating,features):
        self.qid = qid #ID of the query
        self.rating = rating #Rating of this site for this query
        self.features = features #The features of this query-site pair.

    def __str__(self):
        return "Datainstance - qid: "+ str(self.qid)+ ". rating: "+ str(self.rating)+ ". features: "+ str(self.features)


#A class that holds all the data in one of our sets (the training set or the testset)
class dataHolder:

    def __init__(self, dataset):
        self.dataset = self.loadData(dataset)

    def loadData(self,file):
        #Input: A file with the data.
        #Output: A dict mapping each query ID to the relevant documents, like this: dataset[queryID] = [dataInstance1, dataInstance2, ...]
        data = open(file)
        dataset = {}
        for line in data:
            #Extracting all the useful info from the line of data
            lineData = line.split()
            rating = int(lineData[0])
            qid = int(lineData[1].split(':')[1])
            features = []
            for elem in lineData[2:]:
                if '#docid' in elem: #We reached a comment. Line done.
                    break
                features.append(float(elem.split(':')[1]))
            #Creating a new data instance, inserting in the dict.
            di = dataInstance(qid,rating,features)
            if qid in dataset.keys():
                dataset[qid].append(di)
            else:
                dataset[qid]=[di]
        return dataset


def runRanker(trainingset, testset):
    #Dataholders for training and testset
    dhTraining = dataHolder(trainingset)
    dhTesting = dataHolder(testset)

    def buildPatterns(data):
        rating = operator.attrgetter('rating')

        patterns = [(a, b)
                    for queryResults in data.dataset.values()
                    for a, b in itertools.combinations(sorted(queryResults, key=rating, reverse=True), 2)
                    if  a.rating != b.rating]

        # Ensure that data is not ordered
        random.shuffle(patterns)

        return patterns

    trainingPatterns = buildPatterns(dhTraining)
    testPatterns     = buildPatterns(dhTesting)

    runs          = 25
    epochs        = 50
    learning_rate = 0.001
    hidden_nodes  = 10

    results = [[] for _ in range(epochs + 1)]

    for run in range(1, runs + 1):
        #Creating an ANN instance - feel free to experiment with the learning rate (the third parameter).
        nn = Bp.NN(46, hidden_nodes, learning_rate)

        for epoch in range(epochs + 1):
            random.shuffle(trainingPatterns)

            if epoch > 0:
                nn.train(trainingPatterns, iterations=1)

            trainingClassificationRate = 100 * (1 - nn.countMisorderedPairs(trainingPatterns))
            testClassificationRate     = 100 * (1 - nn.countMisorderedPairs(testPatterns))

            results[epoch].append((trainingClassificationRate, testClassificationRate))

            print('INFO run: {}/{} epoch: {}/{} training classification rate: {}% test classification rate: {}%'.format(
                run, runs, epoch, epochs,
                round(trainingClassificationRate, 2),
                round(testClassificationRate, 2)))

    for run in range(1, runs + 1):
        with open('run-{}-epochs-{}-hidden-{}-learning-rate-{}.txt'.format(
            run, epochs, hidden_nodes, learning_rate), 'w') as f:

            for epoch in range(epochs + 1):
                print('{} {} {}'.format(epoch, *results[epoch][run - 1]), file=f)

    with open('runs-{}-epochs-{}-hidden-{}-learning-rate-{}.txt'.format(
        runs, epochs, hidden_nodes, learning_rate), 'w') as f:

        for epoch in range(epochs + 1):
            trainingClassificationRates, testClassificationRates = zip(*results[epoch])

            print('{} {} {} {} {}'.format(
                epoch,
                statistics.mean(trainingClassificationRates),
                statistics.pstdev(trainingClassificationRates),
                statistics.mean(testClassificationRates),
                statistics.pstdev(testClassificationRates)), file=f)

runRanker("train.txt","test.txt")

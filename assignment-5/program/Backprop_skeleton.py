import math
import random

#The transfer function of neurons, g(x)
def logFunc(x):
    return (1.0/(1.0+math.exp(-x)))

#The derivative of the transfer function, g'(x)
def logFuncDerivative(x):
    return logFunc(x) * (1 - logFunc(x))

#Initializes a matrix of all zeros
def makeMatrix(I, J):
    m = []
    for i in range(I):
        m.append([0]*J)
    return m

class NN: #Neural Network
    def __init__(self, numInputs, numHidden, learningRate=0.001):
        #Inputs: number of input and hidden nodes. Assuming a single output node.
        # +1 for bias node: A node with a constant input of 1. Used to shift the transfer function.
        self.numInputs = numInputs + 1
        self.numHidden = numHidden

        # Current activation levels for nodes (in other words, the nodes' output value)
        self.inputActivations  = [1.0] * self.numInputs
        self.hiddenActivations = [1.0] * self.numHidden
        self.outputActivation  = 1.0 #Assuming a single output.
        self.learningRate      = learningRate

        # create weights
        #A matrix with all weights from input layer to hidden layer
        self.weightsInput = makeMatrix(self.numHidden, self.numInputs)
        #A list with all weights from hidden layer to the single output neuron.
        self.weightsOutput = [0] * (self.numHidden + 1) # Assuming single output + bias
        # set them to random vaules
        for j in range(self.numHidden):
            for i in range(self.numInputs):
                self.weightsInput[j][i] = random.uniform(-0.5, 0.5)
        for j in range(self.numHidden + 1): # Bias
            self.weightsOutput[j] = random.uniform(-0.5, 0.5)

        #Data for the backpropagation step in RankNets.
        #For storing the previous activation levels (output levels) of all neurons
        self.prevInputActivations  = [0] * self.numInputs
        self.prevHiddenActivations = [0] * self.numHidden
        self.prevOutputActivation  = 0
        #For storing the previous delta in the output and hidden layer
        self.prevDeltaOutput = 0
        self.prevDeltaHidden = [0] * self.numHidden
        #For storing the current delta in the same layers
        self.deltaOutput = 0
        self.deltaHidden = [0] * self.numHidden

    def propagate(self, inputs):
        if len(inputs) != self.numInputs-1:
            raise ValueError('wrong number of inputs')

        # input activations
        for i in range(self.numInputs-1):
            self.prevInputActivations[i] = self.inputActivations[i]
            self.inputActivations[i]     = inputs[i]
        self.inputActivations[-1] = 1 #Set bias node to -1.

        # hidden activations
        for j in range(self.numHidden):
            sum = 0.0
            for i in range(self.numInputs):
                sum = sum + self.inputActivations[i] * self.weightsInput[j][i]
            self.prevHiddenActivations[j] = self.hiddenActivations[j]
            self.hiddenActivations[j]     = logFunc(sum)

        # output activations
        self.prevOutputActivation=self.outputActivation
        sum = 0.0
        for j in range(self.numHidden):
            sum = sum + self.hiddenActivations[j] * self.weightsOutput[j]
        sum += self.weightsOutput[-1] # Bias

        self.outputActivation = logFunc(sum)
        return self.outputActivation

    def computeOutputDelta(self):
        costFunctionDerivative = logFunc(self.outputActivation - self.prevOutputActivation)
        self.prevDeltaOutput   = logFuncDerivative(self.prevOutputActivation) * costFunctionDerivative
        self.deltaOutput       = logFuncDerivative(self.outputActivation)     * costFunctionDerivative

    def computeHiddenDelta(self):
        deltaOutputDifference = self.prevDeltaOutput - self.deltaOutput

        for i in range(self.numHidden):
            backpropagatedDelta     = self.weightsOutput[i] * deltaOutputDifference
            self.prevDeltaHidden[i] = logFuncDerivative(self.prevHiddenActivations[i]) * backpropagatedDelta
            self.deltaHidden[i]     = logFuncDerivative(self.hiddenActivations[i])     * backpropagatedDelta

    def updateWeights(self):
        for i in range(self.numHidden):
            self.weightsOutput[i] += self.learningRate * (
                self.prevDeltaOutput * self.prevHiddenActivations[i] -
                self.deltaOutput     * self.hiddenActivations[i])

        self.weightsOutput[-1] += self.learningRate * (
            self.prevDeltaOutput - self.deltaOutput) # Bias

        for j in range(self.numHidden):
            for i in range(self.numInputs):
                self.weightsInput[j][i] += self.learningRate * (
                    self.prevDeltaHidden[j] * self.prevInputActivations[i] -
                    self.deltaHidden[j]     * self.inputActivations[i])

    def backpropagate(self):
        self.computeOutputDelta()
        self.computeHiddenDelta()
        self.updateWeights()

    #Prints the network weights
    def weights(self):
        print('Input weights:')
        for j in range(self.numHidden):
            print(self.weightsInput[j])
        print()
        print('Output weights:')
        print(self.weightsOutput)

    def train(self, patterns, iterations=1):
        for _ in range(iterations):
            for a, b in patterns:
                self.propagate(a.features)
                self.propagate(b.features)
                self.backpropagate()

    def countMisorderedPairs(self, patterns):
        numMisses = 0
        numRight  = 0

        activations = {}

        for a, b in patterns:
            activationA = activations.get(a) or activations.setdefault(a, self.propagate(a.features))
            activationB = activations.get(b) or activations.setdefault(b, self.propagate(b.features))

            if activationA > activationB and a.rating > b.rating:
                numRight += 1
            elif activationB > activationA and b.rating > a.rating:
                numRight += 1
            else:
                numMisses += 1

        return numMisses / (numRight + numMisses)

import math
import random
import numpy as np
import matplotlib.pyplot as plt
import time


def dsigmoid(y):
    return 1 * (1.0 - y ** 2)

def InputData(fName):
    f=open(fName)
    text=f.readlines()
    X=[]
    Y=[]
    c=0
    for r in text:
        row=r.strip()
        row=row.replace('\t',' ')
        l=row.split(' ')
        X.append([])
        Y.append([])
        for i in range(2):
            X[c].append(float(l[i]))
        Y[c].append(int(l[2]))
        c+=1
    f.close()
    X=np.array(X)
    Y = np.array(Y)
    return X,Y

class NN:
    def __init__(self, nn_struc):
        # number of input, hidden, and output nodes
        self.Layers = len(nn_struc)
        self.nLayer = []
        for i in range(self.Layers):
            self.nLayer.append(nn_struc[i])

        # activations for nodes
        self.yLayer = []
        for i in range(self.Layers):
            self.yLayer.append([])
        #         for i in range(self.Layers):
        #             self.yLayer.append([1.0]*(self.nLayer[i]))

        # create weights
        self.wi = []
        self.ci = []
        for i in range(0, self.Layers - 1):
            #             self.wi.append(makeMatrix(self.nLayer[i]+1, self.nLayer[i+1]))
            self.wi.append((0.5 - (0.01)) * np.random.rand(self.nLayer[i] + 1, self.nLayer[i + 1]) + (0.01))
            self.ci.append((0.5- (0.01)) * np.random.rand(self.nLayer[i] + 1, self.nLayer[i + 1]) + (0.01))

    def update(self, inputs):
        if len(inputs) != self.nLayer[0] + 1:
            raise ValueError('wrong number of inputs')

        # input activations
        self.yLayer[0] = inputs

        # hidden activations
        for L in range(self.Layers - 2):
            self.yLayer[L + 1] = np.tanh(np.dot(self.yLayer[L], self.wi[L]))
            self.yLayer[L + 1] = np.concatenate((np.ones((1)), self.yLayer[L + 1]), axis=0)
        self.yLayer[self.Layers - 1] = np.tanh(np.dot(self.yLayer[self.Layers - 2], self.wi[self.Layers - 2]))
        return self.yLayer[self.Layers - 1]

    def backPropagate(self, targets, N, M):
        if len(targets) != self.nLayer[self.Layers - 1]:
            raise ValueError('wrong number of target values')

        # calculate error terms for output
        deltas = []
        for i in range(self.Layers - 1):
            deltas.append([])

        error = targets - self.yLayer[self.Layers - 1]
        deltas[self.Layers - 2] = dsigmoid(self.yLayer[self.Layers - 1]) * error

        for i in range(self.Layers - 3, -1, -1):
            #             print(self.yLayer[i+1][1:].shape,self.wi[i+1].shape,deltas[i+1].shape)
            deltas[i] = dsigmoid(self.yLayer[i + 1][1:]) * np.dot(self.wi[i + 1][:-1], deltas[i + 1])

        change = np.dot(self.yLayer[self.Layers - 2].reshape((len(self.yLayer[self.Layers - 2]), 1)),
                        deltas[self.Layers - 2].reshape((1, len(deltas[self.Layers - 2]))))
        self.wi[self.Layers - 2] += N * change + M * self.ci[self.Layers - 2]
        #         self.ci[self.Layers-2] = np.copy(change)
        self.ci[self.Layers - 2] = change
        #         print(self.wi[self.Layers-2])

        for i in range(self.Layers - 3, -1, -1):
            change = np.dot(self.yLayer[i].reshape((len(self.yLayer[i]), 1)), deltas[i].reshape((1, len(deltas[i]))))
            self.wi[i] += N * change + M * self.ci[i]
            #             self.ci[i] = np.copy(change)
            self.ci[i] = change

        # calculate error
        error = np.sum(0.5 * (targets - self.yLayer[self.Layers - 1]) ** 2)
        return error

    def test(self, patterns):
        for p in patterns[0:20]:
            print(p[0], '->', self.update(p[0]), '---', p[1])

    def weights(self):
        print('Input weights:')
        for L in range(self.hLayer - 1):
            print(self.wi[L])
        print()
        print('Output weights:')
        for j in range(self.nh[self.hLayer - 1]):
            print(self.wo[j])

    def train(self, X, Y, iterations=3000, N=0.02, M=0.002):  # 0.03 0.01（1）0.002 0.0001（5）
        # N: learning rate
        # M: momentum factor
        W1 = []
        W2 = []
        W3o = []
        accHist = []
        for i in range(iterations):
            error = 0.0
            accuracy = 0
            tStart = time.time()
            for j in range(len(X)):
                inputs = X[j]
                targets = Y[j]
                out = self.update(inputs)
                if abs(out[0] - targets[0]) > 0.3:
                    accuracy += 1
                error = error + self.backPropagate(targets, N, M)
            acc = accuracy / len(X)
            if i % 100 == 0:
                print('error %-.5f  errPercentage:%.2f' % (error, (acc)))
                tEnd = time.time()
#                 print("####Elapsed %f sec" % (tEnd - tStart))
            if acc < 0.00001:
                print('error %-.5f  errPercentage:%.2f' % (error, (acc)))
                return accHist[:]
            if i % 10 == 0:
                accHist.append(acc)
        return accHist[:]


if __name__ == '__main__':
    n = NN([2, 4,3, 1])
    trainning_data,target = InputData('hw1data.dat')
    print(trainning_data.shape,target.shape)
    trainning_data = np.concatenate((np.ones((trainning_data.shape[0], 1)), trainning_data), axis=1)
    accHist = n.train(trainning_data, target)
    print("done")



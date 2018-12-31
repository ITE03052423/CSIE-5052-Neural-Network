import math
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time


def dsigmoid(y):
    return 1 * (1.0 - y ** 2)


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

        # create weights
        self.wi = []
        self.ci = []
        for i in range(0, self.Layers - 1):
            #             self.wi.append(makeMatrix(self.nLayer[i]+1, self.nLayer[i+1]))
            self.wi.append((1 - (-1)) * np.random.rand(self.nLayer[i] + 1, self.nLayer[i + 1]) + (-1))
            self.ci.append((1 - (-1)) * np.random.rand(self.nLayer[i] + 1, self.nLayer[i + 1]) + (-1))

    def update(self, inputs):
        if len(inputs) != self.nLayer[0] + 1:
            print(len(inputs), self.nLayer[0] + 1)
            raise ValueError('wrong number of inputs')

        # input activations
        self.yLayer[0] = inputs

        # hidden activations
        for L in range(self.Layers - 2):
            self.yLayer[L + 1] = np.tanh(np.dot(self.yLayer[L], self.wi[L]))
            self.yLayer[L + 1] = np.concatenate((np.ones((1)), self.yLayer[L + 1]), axis=0)
        self.yLayer[self.Layers - 1] = np.tanh(np.dot(self.yLayer[self.Layers - 2], self.wi[self.Layers - 2]))
        return self.yLayer[self.Layers - 1]

    def backPropagate2(self, diffC, sameC, N, M):
        x1 = diffC[0]
        x2 = diffC[1]
        self.update(x1)
        y1 = []
        for L in range(self.Layers):
            y1.append(self.yLayer[L].copy())
        self.update(x2)
        y2 = []
        for L in range(self.Layers):
            y2.append(self.yLayer[L].copy())
        # calculate error terms for output
        deltas1 = []
        deltas2 = []
        for i in range(self.Layers - 1):
            deltas1.append([])
            deltas2.append([])

        error = y1[self.Layers - 1] - y2[self.Layers - 1]
        deltas1[self.Layers - 2] = dsigmoid(y1[self.Layers - 1]) * error
        deltas2[self.Layers - 2] = dsigmoid(y2[self.Layers - 1]) * error

        for i in range(self.Layers - 3, -1, -1):
            deltas1[i] = dsigmoid(y1[i + 1][1:]) * np.dot(self.wi[i + 1][:-1], deltas1[i + 1])
            deltas2[i] = dsigmoid(y2[i + 1][1:]) * np.dot(self.wi[i + 1][:-1], deltas2[i + 1])

        change1 = np.dot(y1[self.Layers - 2].reshape((len(y1[self.Layers - 2]), 1)),
                         deltas1[self.Layers - 2].reshape((1, len(deltas1[self.Layers - 2]))))
        change2 = np.dot(y2[self.Layers - 2].reshape((len(y2[self.Layers - 2]), 1)),
                         deltas2[self.Layers - 2].reshape((1, len(deltas2[self.Layers - 2]))))
        self.wi[self.Layers - 2] -= M * (-change1 + change2)

        for i in range(self.Layers - 3, -1, -1):
            change1 = np.dot(y1[i].reshape((len(y1[i]), 1)), deltas1[i].reshape((1, len(deltas1[i]))))
            change2 = np.dot(y2[i].reshape((len(y2[i]), 1)), deltas2[i].reshape((1, len(deltas2[i]))))
            self.wi[i] -= M * (-change1 + change2)
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


class som_perceptron:
    def __init__(self, nMLP):
        # number of input, hidden, and output nodes
        self.MLPLayer = []
        self.MLPLayer.append([nMLP[0]])
        for i in range(nMLP[1]):
            if i != 0:
                self.MLPLayer.append([])
                n = 4
            else:
                n = 3
            for j in range(n):
                self.MLPLayer[i].append(128)

        self.MLP = []
        self.best_NNw = []
        for i in range(len(self.MLPLayer)):
            self.MLP.append(NN(self.MLPLayer[i]))
            self.best_NNw.append([])
            for w_t in self.MLP[i].wi:
                self.best_NNw[i].append(np.copy(w_t))
        self.Y_m = []

        self.min_dist_ever = -1

    def min_dist_diff_class(self, target, pattern, m):
        min_dist = 999999
        y_mt = np.transpose(self.Y_m[m])
        mm, nn = y_mt.shape
        G = np.dot(y_mt.T, y_mt)
        #         print('##G',G.shape)
        H = np.tile(np.diag(G), (nn, 1))
        dist_m = H + H.T - 2 * G
        for i in range(len(target)):
            for j in range(i + 1, len(target)):
                if target[i] != target[j]:
                    if dist_m[i][j] < min_dist:
                        min_dist = dist_m[i][j]
        print("##", m, " min:%.4f" % np.sqrt(min_dist))
        return dist_m

    def get_max_and_min_distance_pair(self, x, target, m, f):

        y_m = []
        for i in range(len(self.Y_m[m])):
            y_m.append(self.MLP[m].update(self.Y_m[m][i]))
        y_mt = np.transpose(y_m)
        mm, nn = y_mt.shape
        G = np.dot(y_mt.T, y_mt)
        H = np.tile(np.diag(G), (nn, 1))
        dist_m = H + H.T - 2 * G

        max_dist_pair = [[], [], [], []]
        min_dist_pair = [[], [], [], []]
        max_dist = -1
        min_dist = 999999
        for i in range(len(target)):
            for j in range(i + 1, len(target)):
                if target[i] == target[j]:
                    continue
                    if dist > max_dist:
                        max_dist = dist
                        max_dist_pair[0] = i
                        max_dist_pair[1] = j
                        max_dist_pair[2] = y1
                        max_dist_pair[3] = y2
                else:
                    if dist_m[i][j] < min_dist:
                        min_dist = dist_m[i][j]
                        min_dist_pair[0] = i
                        min_dist_pair[1] = j
                        min_dist_pair[2] = y_m[i]
                        min_dist_pair[3] = y_m[j]
        if min_dist > self.min_dist_ever:
            for i in range(len(self.MLP[m].wi)):
                self.best_NNw[m][i] = np.copy(self.MLP[m].wi[i])
            self.min_dist_ever = min_dist
        if f == True:
            print('min distance: %.4f  min distance(highest record):%.4f' % (
            np.sqrt(min_dist), np.sqrt(self.min_dist_ever)))
        #             print('min distance:',min_dist)
        return max_dist_pair, min_dist_pair

    def train(self, x, target, eta_att, eta_rep, echo):

        self.echo = echo
        self.eta_att = eta_att
        self.eta_rep = eta_rep

        self.Y_m.append(x)
        #         self.Y_m=np.array(self.Y_m)

        for m in range(len(self.MLP)):
            f = False
            tStart = time.time()
            print('networks structure: ', self.MLPLayer[m])
            for echo_count in range(0, echo):
                f = False
                if echo_count % 200 == 0:
                    f = True
                    print(" echo " + str(echo_count), end="\t")
                max_dist_pair, min_dist_pair = self.get_max_and_min_distance_pair(x, target, m, f)
                self.MLP[m].backPropagate2([self.Y_m[m][min_dist_pair[0]], self.Y_m[m][min_dist_pair[1]]],
                                           [self.Y_m[m][max_dist_pair[0]], self.Y_m[m][max_dist_pair[1]]], eta_att,
                                           eta_rep)
            self.get_max_and_min_distance_pair(x, target, m, True)
            y_m = []
            for i in range(len(self.MLP[m].wi)):
                self.MLP[m].wi[i] = np.copy(self.best_NNw[m][i])
            for i in range(len(self.Y_m[m])):  # Y_m[m]-->preverise layer's outputs  MLP[m]-->nn at this time
                y_m.append(self.MLP[m].update(self.Y_m[m][i]))
            y_m = np.array(y_m)
            y_m = np.concatenate((np.ones((y_m.shape[0], 1)), y_m), axis=1)
            self.Y_m.append(y_m)
            self.min_dist_ever = -1
            tEnd = time.time()


if __name__ == '__main__':

    # build demo_som_perceptron
    trainning_data = pd.read_csv('semeion.data', sep='\s').as_matrix()
    print('training data size: ', trainning_data.shape)
    trainning_data, target_t = np.split(trainning_data[:300], [len(trainning_data[0]) - 10], axis=1)
    target = []
    for i in range(len(target_t)):
        for j in range(len(target_t[0])):
            if target_t[i][j] == 1:
                target.append(j)

    trainning_data = np.concatenate((np.ones((trainning_data.shape[0], 1)), trainning_data), axis=1)
    # trainning demo_som_perceptron with demo_trainning_data
    demo_som_perceptron = som_perceptron([(trainning_data.shape[1] - 1), 1])
    demo_som_perceptron.train(trainning_data, target, 0.00001, 0.0001, 2000)
    with open('./best_W_MLP.dat', 'w+') as f:
        i = 0
        for nn in demo_som_perceptron.best_NNw:
            j = 0
            for l in range(len(nn)):
                print(demo_som_perceptron.MLP[i].wi[j].shape)
                j += 1
                print(nn[l].flatten().shape)
                f.write('{}\n'.format(','.join(list(map(lambda x: str(x), nn[l].flatten())))))
            i += 1




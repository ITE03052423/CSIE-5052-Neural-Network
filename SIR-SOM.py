import math
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

class som_perceptron:
    def __init__(self, nn_struc):
        self.Layers=len(nn_struc)
        self.nLayer=[]
        for i in range(self.Layers):
            self.nLayer.append(nn_struc[i])

        self.Y_m=[]
        self.yLayer=[]
        self.bias=[]
        for i in range(self.Layers):
            self.yLayer.append([1.0]*(self.nLayer[i]))
            if i!=0:
                self.bias.append([1.0]*(self.nLayer[i]))        
        
        self.wi=[]
        for i in range(0,self.Layers-1):
            self.wi.append((1.0-(-1.0))*np.random.rand(self.nLayer[i]+1, self.nLayer[i+1])+(-1))
        
    def min_dist_diff_class(self,target,pattern,m):
        min_dist=np.ones((pattern))*999999
        
        y_mt=np.transpose(self.Y_m[m])
        mm,nn = y_mt.shape
        G = np.dot(y_mt.T, y_mt)
        H = np.tile(np.diag(G), (nn, 1))
        dist_m= H + H.T - 2*G
        for i in range(len(target)):
            for j in range(i+1,len(target)):                
                if target[i]!=target[j]:
                    if dist_m[i][j] < min_dist[target[i]]:
                        min_dist[target[i]]=dist_m[i][j]
                    if dist_m[i][j] < min_dist[target[j]]:
                        min_dist[target[j]]=dist_m[i][j]
        return min_dist
    def update_weight_rand(self,x,target,m,f):
        
        rand_i=np.random.randint(0,len(target),size=(2))
        while(rand_i[0]==rand_i[1]):
            rand_i=np.random.randint(0,len(target),size=(2))
        if f:
            print(rand_i)
        y_p_m = np.dot(self.Y_m[m-1][rand_i[0]],self.wi[m-1])
        y_p_m = np.tanh(y_p_m)
        y_p_m = y_p_m.reshape((len(y_p_m),1))
        y_p_m_1 = self.Y_m[m-1][rand_i[0]].reshape((len(self.Y_m[m-1][rand_i[0]]),1))
        
        y_q_m = np.dot(self.Y_m[m-1][rand_i[1]],self.wi[m-1])
        y_q_m = np.tanh(y_q_m)
        y_q_m = y_q_m.reshape((len(y_q_m),1))
        y_q_m_1 = self.Y_m[m-1][rand_i[1]].reshape((len(self.Y_m[m-1][rand_i[1]]),1))
        
        if target[rand_i[0]]==target[rand_i[1]]:
            delt_wi=self.eta_att*(np.dot(((y_p_m-y_q_m)*(1-y_p_m**2)),np.transpose(y_p_m_1))-np.dot(((y_p_m-y_q_m)*(1-y_q_m**2)),np.transpose(y_q_m_1)))
        else:
            delt_wi=self.eta_rep*(np.dot(((y_p_m-y_q_m)*(1-y_q_m**2)),np.transpose(y_q_m_1))-np.dot(((y_p_m-y_q_m)*(1-y_p_m**2)),np.transpose(y_p_m_1)))
        
        delt_wi=np.transpose(delt_wi)
        self.wi[m-1] -=delt_wi
                
    def update_weight(self,sameC_max_dist_pair,diffC_min_dist_pair,m):
        
#         y_p_m = sameC_max_dist_pair[2].reshape((len(sameC_max_dist_pair[2]),1))
#         y_p_m_1 = self.Y_m[m-1][sameC_max_dist_pair[0]].reshape((1,len(self.Y_m[m-1][sameC_max_dist_pair[0]])))
#         y_q_m = sameC_max_dist_pair[3].reshape((len(sameC_max_dist_pair[3]),1))
#         y_q_m_1 = self.Y_m[m-1][sameC_max_dist_pair[1]].reshape((1,len(self.Y_m[m-1][sameC_max_dist_pair[1]])))

        y_r_m = diffC_min_dist_pair[2].reshape((len(diffC_min_dist_pair[2]),1))
        y_r_m_1 = self.Y_m[m-1][diffC_min_dist_pair[0]].reshape((1,len(self.Y_m[m-1][diffC_min_dist_pair[0]])))
        y_s_m = diffC_min_dist_pair[3].reshape((len(diffC_min_dist_pair[3]),1))
        y_s_m_1 = self.Y_m[m-1][diffC_min_dist_pair[1]].reshape((1,len(self.Y_m[m-1][diffC_min_dist_pair[1]])))
        
        delt_wi=0
#         delt_wi=self.eta_att*(np.dot(((y_p_m-y_q_m)*(1-y_p_m**2)),y_p_m_1)-np.dot(((y_p_m-y_q_m)*(1-y_q_m**2)),y_q_m_1))
        delt_wi+=self.eta_rep*(np.dot(((y_r_m-y_s_m)*(1-y_s_m**2)),y_s_m_1)-np.dot(((y_r_m-y_s_m)*(1-y_r_m**2)),y_r_m_1))
        delt_wi=np.transpose(delt_wi)
        self.wi[m-1] -=delt_wi
        
    def get_max_and_min_distance_pair(self,x,target,m,f):

        y_m=np.dot(self.Y_m[m-1],self.wi[m-1])
        y_m=np.tanh(y_m)
        
        max_dist_pair = [[],[],[],[]]
        min_dist_pair = [[],[],[],[]]
        max_dist = -1
        min_dist = 999999
        
        y_mt=np.transpose(y_m)
        mm,nn = y_mt.shape
        G = np.dot(y_mt.T, y_mt)
        H = np.tile(np.diag(G), (nn, 1))
        dist_m= H + H.T - 2*G
        
        for i in range(len(target)):
            for j in range(i+1,len(target)):                
                if target[i]==target[j]:
                    continue
                    if dist > max_dist:
                        max_dist=dist
                        max_dist_pair[0]=i
                        max_dist_pair[1]=j
                        max_dist_pair[2]=y1
                        max_dist_pair[3]=y2
                else:
                    if dist_m[i][j] < min_dist:
                        min_dist=dist_m[i][j]
                        min_dist_pair[0]=i
                        min_dist_pair[1]=j
                        min_dist_pair[2]=y_m[i]
                        min_dist_pair[3]=y_m[j]
        if f==True:
            print('max distance: ',max_dist, end="\t")
            print('min distance:',min_dist)
        return max_dist_pair,min_dist_pair

    def train(self,x,target,eta_att,eta_rep,echo):

        self.echo = echo
        self.eta_att = eta_att
        self.eta_rep = eta_rep
        
        self.Y_m.append(x)
        print(self.wi[0].shape,x.shape)

        for m in range(1,self.Layers):
            f=False
            tStart=time.time()
            for echo_count in range(0,echo):
                f=False
                if echo_count==echo-1:
                    f=True
                    print("layer " + str(m) + " echo " + str(echo_count), end="\t")

                max_dist_pair,min_dist_pair = self.get_max_and_min_distance_pair(x,target,m,f)
                self.update_weight(max_dist_pair,min_dist_pair,m)
            y_m=np.dot(self.Y_m[m-1],self.wi[m-1])
            y_m=np.tanh(y_m)
            y_m=np.concatenate((np.ones((y_m.shape[0],1)),y_m),axis=1)
            self.Y_m.append(y_m)
            tEnd=time.time()

if __name__ == '__main__':
    
    trainning_data = []
    trainning_data_file = open("hw2pt.dat","r",encoding = "utf-8")
    trainning_data_class_file = open("hw2class.dat","r",encoding = "utf-8")
    
    c=0
    for line in trainning_data_file:
        trainning_data.append([])
        trainning_data[c].append(float(line[:-1].split("	")[0]))
        trainning_data[c].append(float(line[:-1].split("	")[1]))
        c+=1
    
    target=[]
    for line in trainning_data_class_file:
        for single_demo_trainning_data_class_id in range(0,len(line[:-1].split("\t"))):
            target.append(int(line[:-1].split("\t")[single_demo_trainning_data_class_id]))
        break
    trainning_data=np.array(trainning_data)
    target=np.array(target)
    trainning_data=np.concatenate((np.ones((trainning_data.shape[0],1)),trainning_data),axis=1)
        
    plt.figure()    
    for i in range(len(target)):
        if target[i]==1:
            plt.scatter(trainning_data[i][1],trainning_data[i][2], s=15, c='blue', alpha=.5)
        else:
            plt.scatter(trainning_data[i][1],trainning_data[i][2], s=15, c='red', alpha=.5)
    plt.show()
    
    trainning_data_file.close()
    trainning_data_class_file.close()
    
    #build demo_som_perceptron
    demo_som_perceptron = som_perceptron([(trainning_data.shape[1]-1),5,5,5,5,5])
    #trainning demo_som_perceptron with demo_trainning_data
    demo_som_perceptron.train(trainning_data,target,0.000001,0.1,5000)
#     for m in range(demo_som_perceptron.Layers):
#         min_dist=np.sqrt(demo_som_perceptron.min_dist_diff_class(target,10,m))        
#         print(min_dist)
    grid_x = np.copy(trainning_data)
    for i in range(4):
        print(grid_x.shape)
        grid_x = np.dot(grid_x,demo_som_perceptron.wi[i])
        grid_x=np.tanh(grid_x)
        grid_x = np.concatenate((np.ones((grid_x.shape[0],1)),grid_x),axis=1)


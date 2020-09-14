import os
from collections import OrderedDict
import numpy as np
np.set_printoptions(suppress=True)
import matplotlib as mpl
from matplotlib import cm
import matplotlib.pyplot as plt
from time import time
from copy import copy

class designer():

    def __init__(self,ff,weight,method='D'):
        '''
        input:
        ------
        ff: 2-D array. Rows represent points in the pool; columns represent parameters
            involved in the direvative.
        weight: 1-D array. Its length equals to the total number of points in the pool,
                or the numnber of rows of 'ff'.
        method: The criterion used for the optimization, default is D-optimal method.
        '''
        self.ff = ff
        self.m = ff.shape[1] # number of parameters
        self.weight = weight
        self.N_candidates = np.sum(weight!=0)
        self.method = method
        self.d = 0 # sensitivity function
        self.d_max = 0 # initialize the maximum of sensitivity.
        self.id_minimax = None
        self.M = 0 # information matrix
        self.M_inv = self.M # information matrix inverse
        self.psi_iter = [] # all the optimal criteria over the iterative procedure
        self.phi_iter = [] # all the sensitivity function ove the iterative procedure
        self.weight_iter = []


    def cal_criterion(self,local=False):

        self.M = 0
        for i,f in enumerate(self.ff):
            self.M += self.weight[i] * np.outer(f,f)
        self.M_inv = np.linalg.inv(self.M)

        if self.method == 'D':
            self.d = np.array([f @ self.M_inv @ f for f in self.ff])

        if local==False:
            self.id_minimax = np.argmax(self.d)
            self.d_max = self.d[self.id_minimax]

        else:
            self.id_minimax = np.argmin(np.ma.array(self.d,mask=(self.weight==0)))


    def collect(self):
        self.psi_iter.append(np.linalg.det(self.M_inv))
        self.phi_iter.append(self.d_max)
        self.weight_iter.append(self.weight)


    def update_design(self, alpha, action='add'):

        if action == 'add':
            alpha_s = alpha
        elif action == 'remove':
            p_s = self.weight[self.id_minimax]
            alpha_s = -min(alpha, p_s/(1-p_s))
        else:
            print("Design not updated")
            return 1

        self.weight = self.weight * (1-alpha_s) # reduce current design by alpha
        self.weight[self.id_minimax] += alpha_s # add the new point weighted by alpha
        self.weight = self.weight / sum(self.weight) # renormalize weight

        return 0


    def optimize(self,verbose=False,delta=1e-5,max_steps=1e6,remove=False):

        if delta == None:
            threshold = 0 # no limit on "d_max"
        else:
            threshold = self.m / (1-delta)

        # the stop condition: either maximum steps or threshold met.
        stop = lambda s: s >= max_steps or self.d_max <= threshold

        step = 0
        self.cal_criterion(local=False)
        self.collect()
        while not stop(step):
            step += 1
            alpha = 1 / (1+step+self.N_candidates) # step length

            self.cal_criterion(local=False)
            if self.update_design(alpha,action='add'):
                break

            if remove == True:
                self.cal_criterion(local=True)
                if self.update_design(alpha,action='remove'):
                    break

            self.collect()

        if verbose:
            print('Iteration steps: {}'.format(step))
            print('criterion: {:.3f}'.format(self.m/self.d_max))

    def timer():
        pass

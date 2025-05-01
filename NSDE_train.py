#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 29 12:31:04 2025

@author: lukesnow
"""

################################
# Intitial imports and plot def:
################################
import torch
from torch import nn
import numpy as np
import random
import math

import time

import os
import sys


module_path = os.path.abspath(os.path.join('..'))

if module_path not in sys.path:
    sys.path.append(module_path)

#matplotlib inline
import matplotlib.pyplot as plt

import torchsde

def plot(ts, samples, xlabel, ylabel, title=''):
    ts = ts.cpu()
    samples = samples.squeeze().t().cpu()
    plt.figure()
    for i, sample in enumerate(samples):
        plt.plot(ts, sample, marker='x', label=f'sample {i}')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.show()

batch_size, state_size = 15, 1
brownian_size = state_size

plt.close('all')


################################
# define time varying linear nn layer:
################################
class TimeVaryingLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super(TimeVaryingLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        # Define a parameter or module to generate weights dynamically
        self.weight_generator = nn.Linear(1, in_features * out_features) # Example: using time as input

    def forward(self, x, t):
        # Generate weights based on time (t)
        dynamic_weights = self.weight_generator(t.unsqueeze(0).unsqueeze(0))
        dynamic_weights = dynamic_weights.view(self.out_features, self.in_features)

        # Perform the forward pass with the dynamic weights
        return torch.matmul(x, dynamic_weights.T)
    
################################
# define SDE structure
################################
class SDE(nn.Module):

    noise_type = "diagonal"
    sde_type = "ito"

    def __init__(self):
        super().__init__()
        self.tvl1 = TimeVaryingLinear(state_size, state_size) #input_size

        self.tvl2 = TimeVaryingLinear(state_size, state_size) #input_size
        
        self.dfc1 = torch.nn.Linear(state_size, 
                                     state_size)
        self.dfc2 = torch.nn.Linear(state_size, 
                                     state_size)# * brownian_size) 

    # Drift
    def f(self,t,x):
        # parametrized NN:
        m = nn.ELU()
        x = m(self.tvl1(x, t))
        x = self.tvl2(x,t) # no m
     #   x = x * self.tfc1(t) + self.tfc2(t) # affine trainable time-varying drift [?: HOW TO]
        return x 

    # Diffusion
    def g(self,t,x):
        # parametrized NN:
        m = nn.ELU()
        x = m(self.dfc1(x))
        x = self.dfc2(x) #no m
        return x 
    
    
##############################
# Pre-training SDE behavior
##############################

y0 = torch.full(size=(batch_size, state_size), fill_value=0.1)

t_size = 100
sde = SDE()
ts = torch.linspace(0, 1, t_size)
y0 = torch.full(size=(batch_size, state_size), fill_value=0.1)

with torch.no_grad():
    ys = torchsde.sdeint(sde, y0, ts, method='euler')  # (t_size, batch_size, state_size) = (100, 3, 1).


plot(ts, ys, xlabel='$t$', ylabel='$Y_t$')


##############################
# SDE Training
##############################


rng = np.arange(0, 1.0, 0.01)
fcn = rng**3
#fcn = np.sin(2*np.pi*x)

step_size = 0.1
len = 500 #training iterations
ls = torch.linspace(0, 1, len)
#save = torch.empty(ys.size()[0],4,1)
save = torch.empty(len,4,1)
ls = torch.empty(len)
for k in range(len):
    step_size = 0.1
    ys = torchsde.sdeint(sde, y0, ts, method='euler')
    
    #y_final = ys[-1]
    target = np.zeros(ys.size())
    #target[:,1] = y
    ### More efficient way to do this assignment:
    target[:] = fcn[:, None, None] 
    
    #loss = ((target - y_final) ** 2).sum(dim=1).mean(dim=0) 
    loss = (torch.sub(torch.tensor(target),ys)**2).sum(dim=1).mean(dim=0)
    ls[k] = loss.clone().detach()
    
    t0 = time.time()
    grad = torch.autograd.grad(loss, inputs =[sde.tvl1.weight_generator.weight,sde.tvl2.weight_generator.weight,sde.dfc1.weight,sde.dfc2.weight]) 
    td = time.time() - t0 # time to compute gradient
    # ^ time_varying_linear.weight_generator -> mfc1

    g_mfc1 = grad[0].detach().item()
    save[k][0] = sde.tvl1.weight_generator.weight.clone().detach()
    sde.tvl1.weight_generator.weight = nn.Parameter(sde.tvl1.weight_generator.weight.clone().detach() -  step_size*g_mfc1, requires_grad=True)  


    g_mfc2 = grad[1].detach().item()
    save[k][1] = sde.tvl2.weight_generator.weight.clone().detach()
    sde.tvl2.weight_generator.weight = nn.Parameter(sde.tvl2.weight_generator.weight - step_size*g_mfc2, requires_grad=True) 
    
    
    g_dfc1 = grad[2].detach().item()
    save[k][2] = sde.dfc1.weight.clone().detach()
    sde.dfc1.weight = nn.Parameter(sde.dfc1.weight.clone().detach() - step_size*g_dfc1, requires_grad=True)  
    
    
    g_dfc2 = grad[3].detach().item()
    save[k][3] = sde.dfc2.weight.clone().detach()
    sde.dfc2.weight = nn.Parameter(sde.dfc2.weight.clone().detach() - step_size*g_dfc2, requires_grad=True)  
    
##############################
# Plot training optimization:
##############################

ti  = torch.linspace(0, 1, len)
plot(ti, save, xlabel='$training iteration$', ylabel='$weight value$')
plt.plot(ti,ls)
plt.show()

###############################
# After training behavior:
###############################


batch_size, state_size, t_size = 5, 1, 100
#sde = SDE()
ts = torch.linspace(0, 1, t_size)
y0 = torch.full(size=(batch_size, state_size), fill_value=0.1)

with torch.no_grad():
    ys = torchsde.sdeint(sde, y0, ts, method='euler')  # (t_size, batch_size, state_size) = (100, 3, 1).


plt.plot(ts.tolist(), fcn)
plt.show()
plot(ts, ys, xlabel='$t$', ylabel='$Y_t$')





















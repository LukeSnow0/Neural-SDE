#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 14 14:32:54 2025

@author: lukesnow
"""


######################################################
# Order-5 Cubature Path Base Initialization
######################################################


# %%
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import svd
import math

import torchdiffeq


def path_1(t):
    if t <= 1/3:
        return -math.sqrt(3)*(1/2*(4-math.sqrt(22))*t) 
    elif 1/3 < t <= 2/3:
        return -math.sqrt(3)*(1/6 * (4-math.sqrt(22)) + (-1 + math.sqrt(22))*(t-1/3))
    else:
        return -math.sqrt(3)*(1/6*(2 + math.sqrt(22)) + 1/2 * (4 - math.sqrt(22))*(t-2/3))

def path_2(t):
    if t <= 1/3:
        return math.sqrt(3)*(1/2*(4-math.sqrt(22))*t) 
    elif 1/3 < t <= 2/3:
        return math.sqrt(3)*(1/6 * (4-math.sqrt(22)) + (-1 + math.sqrt(22))*(t-1/3))
    else:
        return math.sqrt(3)*(1/6*(2 + math.sqrt(22)) + 1/2 * (4 - math.sqrt(22))*(t-2/3))


def path_3(t): 
    return 0

"""
def path(t,x0):
    if t <= 1/3:
        return x0 + 1/2*(4-math.sqrt(22))*t
    elif 1/3 < t <= 2/3:
        return x0 + 1/6*(4-math.sqrt(22)) + (-1 + math.sqrt(22))*(t-1/3)
    else:
        return x0 + 1/6*(2 + math.sqrt(22)) + 1/2*(4 - math.sqrt(22))*(t-2/3)
"""

"""
# Example usage
x_vals = np.linspace(0, 1, 100)
#p1_vals = [-math.sqrt(3)*path(x,0) for x in x_vals]
#p2_vals = [math.sqrt(3)*path(x,0) for x in x_vals]

p1_vals = [path_1(x) for x in x_vals]
p2_vals = [path_2(x) for x in x_vals]
p3_vals = [path_3(x) for x in x_vals]

# Plotting the function
plt.figure(figsize=(8, 6))
plt.plot(x_vals, p1_vals)
plt.plot(x_vals, p2_vals)
plt.plot(x_vals, p3_vals)
plt.xlabel('x')
plt.ylabel('f(x)')
plt.grid(True)
plt.show()
"""

######################################################
# Stieltjes ODE solver
######################################################


def solve_stieltjes_ode(v1,v2, g, t_span, y0, d_num):
       """
       Solves a Stieltjes ODE dy(t) = \sum_{i=1}^2 vi(t, y(t)) dg(t), generalize for higher Brownian dimensions 
       using the forward Euler method.

       Args:
           f (function): Function defining the ODE, f(t, y).
           g (function): Integrator function, g(t).
           t_span (tuple): Time span (t_start, t_end).
           y0 (float): Initial condition, y(t_start).

       Returns:
           tuple: Arrays of time points and solution values.
       """
       t_start, t_end = t_span
       t_points = torch.linspace(t_start, t_end, d_num) # Adjust the number of points as needed
       y_values = torch.zeros((len(t_points)))
       y_values[0] = y0

       for i in range(len(t_points) - 2):
           dt = t_points[i+1] - t_points[i]
           dg = g(t_points[i+1]) - g(t_points[i])
           y_values[i+1] = y_values[i] + v1(t_points[i], y_values[i].unsqueeze(0).unsqueeze(0)) * dt + v2(t_points[i], y_values[i].unsqueeze(0).unsqueeze(0)) * dg
       
       return t_points, y_values
   
### https://stackoverflow.com/questions/49852455/how-to-find-the-null-space-of-a-matrix-in-python-using-numpy
def nullspace(A, atol=1e-13, rtol=0):
    A = np.atleast_2d(A)
    u, s, vh = svd(A)
    tol = max(atol, rtol * s[0])
    nnz = (s >= tol).sum()
    ns = vh[nnz:].conj().T
    return ns

######################################################
# Vector Field Initialization
######################################################


import torch
from torch import nn
import torchsde
import os
import sys


module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)


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

state_size, brownian_size = 1, 1


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
      #  dynamic_weights = self.weight_generator(t)
      #  dynamic_weights = self.weight_generator(torch.tensor(t).unsqueeze(0).unsqueeze(0))
        dynamic_weights = dynamic_weights.view(self.out_features, self.in_features)

        # Perform the forward pass with the dynamic weights
        return torch.matmul(x, dynamic_weights.T)


class SDE(nn.Module):

    noise_type = "diagonal"
    sde_type = "stratonovich"

    def __init__(self):
        super().__init__()
        self.tvl1 = TimeVaryingLinear(state_size, state_size) #input_size
        self.tvl2 = TimeVaryingLinear(state_size, state_size) #input_size
        self.tvl3 = TimeVaryingLinear(state_size, state_size)

        self.dfc1 = torch.nn.Linear(state_size, 
                                     state_size)
        self.dfc2 = torch.nn.Linear(state_size, 
                                     state_size)# * brownian_size)
    # Drift
    def f(self,t,x):
        # parametrized NN:
        m = nn.ELU()
        x = m(self.tvl1(x,t))
        x = m(self.tvl2(x,t))
        x = self.tvl3(x,t) # no m
        return x 
        
        
    # Diffusion
    def g(self,t,x):
        # parametrized NN:
        m = nn.ELU()
        x = m(self.dfc1(x))
        x = self.dfc2(x) #no 
        return x 
        
        
    
    
######################################################
# MC high-accuracy expected loss calculation (oracle)
######################################################

"""
batch_space = np.linspace(10,10000,10)
l_save = np.empty(10)
rng = np.arange(0, 1.0, 0.001)


sde = SDE() ### This is where SDE vector field NN parameters are initialized, fixed

#####: MC looping (to see convergence)

for i in range(10):
    #loss = 0
    #for m in range(int(batch_space[i])):
        
    y0 = torch.full(size=(int(batch_space[i]), state_size), fill_value=0.1)
    fcn = rng**2
    
    t_size = 1000
    ts = torch.linspace(0, 1, t_size)
    y0 = torch.full(size=(int(batch_space[i]), state_size), fill_value=0.1)
    
    with torch.no_grad():
        ys = torchsde.sdeint(sde, y0, ts, method='midpoint') #'euler'  # (t_size, batch_size, state_size) = (100, 3, 1).
    
    target = np.zeros(ys.size())
    target[:] = fcn[:, None, None] 
    
    #loss = ((target - y_final) ** 2).sum(dim=1).mean(dim=0) 
    #loss = (torch.sub(torch.tensor(target),ys)**2).sum(dim=1).mean(dim=0)
    loss = (((torch.sub(torch.tensor(target),ys)**2).sum(dim=0))*0.01).mean(dim=0)

    l_save[i] = loss
    #l_save[i] = loss[0] / batch_space[i]


"""

##### Direct MC evaluation

#b_size = 1
batch_size = 1000
#l_save = np.empty(b_size)
rng = np.arange(0, 1.0, 0.001)

### Should converge to E[L_{data}(X^{\theta})]: try higher batch sizes or direct Monte-Carlo eval.

sde = SDE()

loss = 0

for m in range(int(batch_size)):
    
    y0 = torch.full(size=(1, state_size), fill_value=0.1)
    fcn = rng**2
    
    t_size = 1000
    ts = torch.linspace(0, 1, t_size)
    y0 = torch.full(size=(1, state_size), fill_value=0.1)
    
    with torch.no_grad():
        ys = torchsde.sdeint(sde, y0, ts, method='midpoint') #'euler'  # (t_size, batch_size, state_size) = (100, 3, 1).
    
    target = np.zeros(ys.size())
    target[:] = fcn[:, None, None] 
    
    #loss = ((target - y_final) ** 2).sum(dim=1).mean(dim=0) 
    #loss = (torch.sub(torch.tensor(target),ys)**2).sum(dim=1).mean(dim=0)
    # loss = (((torch.sub(torch.tensor(target),ys)**2).sum(dim=0))*0.001).mean(dim=0)
    loss += (((torch.sub(torch.tensor(target),ys)**2).sum(dim=0))*0.001)

    #  l_save[i] = loss
    if (m%100==0):
        print(m/100)
        print(loss[0] / m) # normalized loss
      
l_save = loss[0] / batch_size
    

"""
plt.plot(batch_space[1:],l_save[1:])
plt.show()
"""

#print(loss)


"""
plt.plot(batch_space[1:],l_save[1:])
plt.show()
"""


"""# %% Test Cell:
x_space = np.linspace(-100,100,1000)
f = np.zeros(1000)
for i in range(len(x_space)):
    f[i] = sde.g(1,torch.tensor(x_space[i]))
plt.plot(x_space,f)
plt.show()
"""

# %% Cell 2

######################################################
# Order 5 Cubature Path Construction
######################################################

disc_order = np.array([1,2,3,4,5,6,7,8,9,10])
stat = np.empty(len(disc_order))

ind = 0 # indicator

for order in range(len(disc_order)): # k num of discretization steps

  #  print(cub_order[order])
    
    k = disc_order[order]
    
    t_d = np.zeros(k+1) # discretization time values
    g = 2 # gamma parameter
    for _ in range(k+1):
        t_d[_] = (1-(1-_/k)**g) 
        
    # initialize first-step paths:
    dtt = 0
    nv = 100
    dt = t_d[1] - t_d[0]
    dtn = int(dt*100)
    x_loc = np.linspace(0,dt,dtn)
    
    p = np.zeros((3,dtn))
    
    w = np.zeros(p.shape[0])
    
    dtt += dtn
    
    #p[0,:] = [-math.sqrt(3)*math.sqrt(t_d[1]-t_d[0])*path(x/(t_d[1]-t_d[0]),0) for x in x_vals]
    #p[1,:] = [math.sqrt(3)*math.sqrt(t_d[1]-t_d[0])*path(x/(t_d[1]-t_d[0]),0) for x in x_vals]
    p[0,:] = [math.sqrt(t_d[1]-t_d[0])*path_1(x/(t_d[1]-t_d[0])) for x in x_loc]
    p[1,:] = [path_3(x) for x in x_loc]
    p[2,:] = [math.sqrt(t_d[1]-t_d[0])*path_2(x/(t_d[1]-t_d[0])) for x in x_loc]
    #n_paths  = p.shape[0]
    
    w[0] = 1/6
    w[1] = 2/3
    w[2] = 1/6
    
    
    for _ in range(1,k):
      #  x_loc = np.linspace(t_d[_],t_d[_+1],nv)
        dt = t_d[_+1]-t_d[_]
        dtn = int(dt*100)
        x_loc = np.linspace(0,dt,dtn)
        
        rows = p.shape[0]
        q = p
        p = np.empty((rows*3,q.shape[1]))
    
        w_ = w
        w = np.empty((rows*3))
        
        ## copy path rows
        for i in range(rows*3):
            if i%3 == 0:
                p[i,:] = q[int(i/3),:]
            else:
                p[i,:] = p[i-1,:]
    
        ## copy paths to form new columns, to fill
        p = np.concatenate((p,p[:,0:dtn]),axis=1)
    
        ### add weight vector
        
        for n in range(p.shape[0]):
            # compute path appendages:
            x0 = p[n,dtt-1]
            p[n,dtt:] = x0*np.ones(len(p[n,dtt:]))
            if n%3 == 0: 
               # p[n,nv*_:] = [-math.sqrt(3)*math.sqrt(t_d[_+1]-t_d[_])*path(x/(t_d[_+1]-t_d[_]),x0) for x in x_loc]
                p[n,dtt:] += [math.sqrt(t_d[_+1]-t_d[_])*path_1(x/(t_d[_+1]-t_d[_])) for x in x_loc]
                w[n] = w_[n//3]*(1/6)
            elif (n-1)%3 ==0 : 
                p[n,dtt:] += [path_3(x) for x in x_loc]
                w[n] = w_[n//3]*(2/3)
            else:
                # p[n,nv*_:] = [math.sqrt(3)*math.sqrt(t_d[_+1]-t_d[_])*path(x/(t_d[_+1]-t_d[_]),x0) for x in x_loc]   
                p[n,dtt:] += [math.sqrt(t_d[_+1]-t_d[_])*path_2(x/(t_d[_+1]-t_d[_])) for x in x_loc]
                w[n] = w_[n//3]*(1/6)
    
        dtt += dtn
        
      #  print(w[0])
        
        
        ######## High-Order Recombination ############
        ##############################################

        pts = p[:,-1] #end-points of all paths
        #if (_ == k-1):
        print(f'pts: {np.sum(pts)}')
        # print(f'pts: {np.sum(pts)}')
        
        
        # choose localization radius u_i (u_{_+1}):
        # u = dt**(5/(2*g)) # s_i^{p*/2gamma} (assume p* <= 5) -> generalize
        u = dt
        thres = 3 # r+1
        
        # partition into regions U_j:
        mx =  np.max(pts)
        mn = np.min(pts)
       # prt = int((mx-mn) / u)
        prt = np.linspace(mn,mx,int((mx-mn)/u))
    
       # ref_t = np.max(pts)
       # ref_b = ref_t
        pts_loc = 0.5*np.ones((len(prt),1))
       # for j in range(prt):
        j=0
        for j in range(len(prt)-1):
           # if (j >= 1):
           #     ref_t = ref_b
           # ref_b = ref_b - u
            find = np.where(np.logical_and(pts >= prt[j], pts < prt[j+1]))[0]
            if (len(find) > thres):
                ind = 1 #indicator
                # construct A_0 matrix:
                for pd in range(thres): #polynomial degree <= r+1
                    if (pd==0):
                        A = np.ones(len(find))
                    else:
                        A = np.vstack((A,pts[find]**pd))
                # find kernel basis via svd:
                kern = nullspace(A) # kernel basis
                # nullspace dimension:
                null_dim = len(kern[0,:])
                w_j = w[find] 
                
                e = np.zeros(null_dim)

                for i in range(null_dim):
                   # determine node to be eliminated:
                   frac = w_j / kern[:,0] 
                   frac = frac.tolist()
                   alpha = min([i for i in frac if i > 0]) # min pos. value
                  # alpha = np.min(frac)
                   e[i] = frac.index(alpha)
    
                   # remove e[i]'th column:
                   A = np.delete(A, int(e[i]), 1)
    
                   # update probabilities and kernel basis:
                   w_j_ = w_j # placehold
                   w_j = np.zeros(len(w_j_)-1)
                   #d = np.zeros(len(w_j))
                   
                   for l in range(len(w_j)):
                       if (l < e[i]):
                           w_j[l] = w_j_[l] - alpha*(kern[l,0])
                       else:
                           w_j[l] = w_j_[l+1] - alpha*(kern[l+1,0]) 
    
                   kern_ = kern # placehold to reduce by one column
                   kern = np.zeros((kern_.shape[0],kern_.shape[1]-1))
                   d = np.zeros(kern.shape[1])
                   for l in range(kern.shape[1]):
                       d[l] = kern_[int(e[i]),l+1] / kern_[int(e[i]),0]
                       kern[:,l] = kern_[:,l+1] - d[l]*kern_[:,0]
                       
                   kern = np.delete(kern,int(e[i]),0)
                   
                   # delete find[e[i]]'th row of p
                   p = np.delete(p,find[int(e[i])],0)
                   pts = p[:,-1]
                   
                   
                   w = np.delete(w,find[int(e[i])],0)
    
                   find = np.delete(find,int(e[i]),0)
                   for k in range(len(find)):
                       if (k >= e[i]):
                           find[k] = find[k] - 1
        
                w[find] = w_j
                j += 1
      #  print(w[0])
      
    
    
    x_vals = np.linspace(0,1,dtt)
    
    
    plt.figure(figsize=(8, 6))
    for n in range(p.shape[0]):
        plt.plot(x_vals, p[n,:])
    #plt.plot(x_vals, p[0,:])
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.grid(True)
    plt.show() 
    
    
    ######################################################
    # ODE Cubature Solve
    ######################################################
    
    
    y0 = 0.1
    t_span = (0,1)
    v = np.zeros((p.shape[0],p.shape[1]-1))
    v_ = np.zeros(p.shape[0])
    for n in range(p.shape[0]):
        
        om = lambda t: p[n,int(np.floor(t*dtt))]
        
        time, vals = solve_stieltjes_ode(sde.f, sde.g, om, t_span, y0, p.shape[1])
    
        v[n,:] = ((time[:-1]**2 - vals[:-1])**2).detach().numpy() # for specific loss function (t^2 - .)^2
        
        ## scale by weights
        v[n,:] *= w[n]
    
        v_[n] = np.sum(v[n,:])*(1/p.shape[1])
    
    
    stat[order] = np.abs(np.sum(v_) - l_save)
    
    print(f'error: {stat[order]}')
    print(f'ind: {ind}')
   # print(p.shape[0])
    
plt.plot(disc_order, stat)

#plt.plot(cub_order,cub_order**(-g))
plt.show()
 
######### ^ should be similar to 'loss'




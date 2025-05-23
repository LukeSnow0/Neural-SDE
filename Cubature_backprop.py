#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 13 12:37:49 2025

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
from torchdiffeq import odeint
from torchdiffeq import odeint_adjoint

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
       y_values = torch.zeros((t_points.shape[0])) #len(t_points)
       y_values[0] = y0

       for i in range(t_points.shape[0] - 2):
           #dt = t_points[i+1] - t_points[i]
           #dg = g(t_points[i+1]) - g(t_points[i])
           y_values[i+1] = y_values[i] + v1(t_points[i], y_values[i].unsqueeze(0).unsqueeze(0)) * (t_points[i+1] - t_points[i])  + v2(t_points[i], y_values[i].unsqueeze(0).unsqueeze(0)) * (g(t_points[i+1]) - g(t_points[i]))
       
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
        self.double()
        self.weight_generator.double()
        
    def forward(self, x, t):
        # Generate weights based on time (t)
        dynamic_weights = self.weight_generator(t.unsqueeze(0).unsqueeze(0).to(torch.float64)) #.to(torch.float64)
      #  dynamic_weights = self.weight_generator(t)
      #  dynamic_weights = self.weight_generator(torch.tensor(t).unsqueeze(0).unsqueeze(0))
        dynamic_weights = dynamic_weights.view(self.out_features, self.in_features)
        dynamic_weights.to(torch.float64)
        
        # Perform the forward pass with the dynamic weights
        return torch.matmul(x.to(torch.float64), dynamic_weights.T) #


class SDE(nn.Module):

    noise_type = "diagonal"
    sde_type = "stratonovich"

    def __init__(self):
        super().__init__()
        self.tvl1 = TimeVaryingLinear(state_size, state_size) #input_size
        self.tvl2 = TimeVaryingLinear(state_size, state_size) #input_size
        self.tvl3 = TimeVaryingLinear(state_size, state_size)

        self.dfc1 = torch.nn.Linear(state_size, 
                                     state_size).double()
        self.dfc2 = torch.nn.Linear(state_size, 
                                     state_size).double()# * brownian_size)
       # self.double()
        
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
        x = m(self.dfc1(x.double()))
        x = self.dfc2(x.double()) #no 
        return x 


sde = SDE() #define sde; initializes random NN weights

"""
######################################################
# MC high-accuracy expected loss calculation (oracle)
######################################################

#b_size = 1
batch_size = 2000
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

# %%

######################################################
# Order 5 Cubature Path Construction: Pre-Processing (without recom)
######################################################

disc_order = 3 #np.array([2,3,4,5,6,7,8])


#  print(cub_order[order])

k = disc_order

t_d = np.zeros(k+1) # discretization time values
g = 2 # gamma parameter
for _ in range(k+1):
    t_d[_] = (1-(1-_/k)**g) 
    
# initialize first-step paths:
dtt = 0
nv = 100
dt = t_d[1] - t_d[0]
dtn = int(dt*200)
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
    dtn = int(dt*200)
    x_loc = np.linspace(dt/dtn,dt,dtn) #np.linspace(0,dt,dtn)
    
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

    pts = p[:,-1]
   # print(f'before pts: {np.sum(pts)}')
    
    for n in range(p.shape[0]):
        # compute path appendages:
        x0 = p[n,dtt-1]
        p[n,dtt:] = x0*np.ones(p[n,dtt:].shape[0]) #len(p[n,dtt:])
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
       # if order == 4 and _ == 5:
        pts = p[:,-1]

    dtt += dtn

    """
    ######## High-Order Recombination ############
    ##############################################

    pts = p[:,-1] #end-points of all paths
    iind = 0
    for i in range(len(pts)):
        if len(np.where(pts == -pts[i])[0])==0:
            iind += 1
            
    # choose localization radius u_i (u_{_+1}):
    # u = dt**(5/(2*g)) # s_i^{p*/2gamma} (assume p* <= 5) -> generalize
    u = dt
    thres = 15 # r+1
    
    # partition into regions U_j:
    mx =  np.max(pts)
    mn = np.min(pts)
    #prt = np.linspace(mn,mx,int((mx-mn)/u))
    prt = np.linspace(0,mx,int(mx/u))
    
   # ref_t = np.max(pts)
   # ref_b = ref_t
    pts_loc = 0.5*np.ones((len(prt),1))
   # for j in range(prt):
    j=0
    for j in range(len(prt)-1):
       # if (j >= 1):
       #     ref_t = ref_b
       # ref_b = ref_b - u
        find = np.where(np.logical_and(pts > prt[j], pts <= prt[j+1]))[0]
       # find_ = np.where(np.logical_and(pts <= -prt[j], pts > -prt[j+1]))[0] #negative symmetric indices
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
                
          #  e = np.zeros(null_dim)

            for i in range(null_dim):
               # determine node to be eliminated:
               frac = w_j / kern[:,0] 
               frac = frac.tolist()
               alpha = min([m for m in frac if m > 0]) # min pos. value
              # alpha = np.min(frac)
               e = frac.index(alpha)

               # remove e[i]'th column:
               A = np.delete(A, int(e), 1)

               # update probabilities and kernel basis:
               w_j_ = w_j # placehold
               w_j = np.zeros(len(w_j_)-1)
               #d = np.zeros(len(w_j))
               
               for l in range(len(w_j)):
                   if (l < e):
                       w_j[l] = w_j_[l] - alpha*(kern[l,0])
                   else:
                       w_j[l] = w_j_[l+1] - alpha*(kern[l+1,0]) 

               kern_ = kern # placehold to reduce by one column
               kern = np.zeros((kern_.shape[0],kern_.shape[1]-1))
               d = np.zeros(kern.shape[1])
               for l in range(kern.shape[1]):
                   d[l] = kern_[int(e),l+1] / kern_[int(e),0]
                   kern[:,l] = kern_[:,l+1] - d[l]*kern_[:,0]
                   
               kern = np.delete(kern,int(e),0)
               
               # delete find[e[i]]'th row of p
               neg_ind = np.where(pts == -pts[find[int(e)]])[0][0]
               p = np.delete(p,[find[int(e)],neg_ind],0)
            #   p = np.delete(p,neg_ind,0)
               
             #  print(len(np.where(pts == -pts[find[int(e[i])]])[0]))
               
               w = np.delete(w,[find[int(e)],neg_ind],0)
            #   w = np.delete(w,neg_ind,0)

               #print(pts[find[int(e)]])
               #print(pts[neg_ind])
                
               pts = p[:,-1]
                
               find = np.delete(find,int(e),0)
               for k in range(len(find)):
                   if (k >= e):
                       find[k] = find[k] - 1
    
            w[find] = w_j
            j += 1
  #  print(w[0])
  """

x_vals = np.linspace(0,1,dtt)
    
plt.figure(figsize=(8, 6))
for n in range(p.shape[0]):
    plt.plot(x_vals, p[n,:])
#plt.plot(x_vals, p[0,:])
plt.xlabel('x')
plt.ylabel('f(x)')
plt.grid(True)
plt.show() 




"""
class NeuralODE(nn.Module):
    def __init__(self, t, state):
        super().__init__()
        self.initial_pos = nn.Parameter(torch.tensor([0]))
      #  self.initial_aug = nn.Parameter(torch.zeros(aug_dim))
        self.odefunc = sde.f(t,state) + sde.g(t,state)*(p[0,t*len(p[0,:])]-p[0,t*len(p[0,:])])
   

        def init(m):
            if isinstance(m, nn.Linear):
                std = 1.0 / math.sqrt(m.weight.size(1))
                m.weight.data.uniform_(-2.0 * std, 2.0 * std)
                m.bias.data.zero_()

        self.odefunc.apply(init)

    def forward(self, t, state):
        return self.odefunc(t,state)

    def simulate(self, times):
        x0 = self.initial_pos
        solution = odeint(self, x0, times, atol=1e-8, rtol=1e-8, method="dopri5")
        trajectory = solution[:, 0]
        return trajectory, []
"""

######################################################
# ODE Cubature Solve
######################################################
"""
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

    v_[n] = np.sum(v[n,:])*(1/p.shape[1]) #approximates E[\int f(t,X_t)dt]

cub_loss = np.sum(v_)
#stat[order] = np.abs(cub_loss - l_save)
"""

# %%
##############################
# SDE Training
##############################
#from autograd import grad
#from torchdiffeq import odeint_adjoint

class vectopath(nn.Module):
    def __init__(self):
        super().__init__()
        self.t_points = torch.linspace(0,1,p.shape[1])
        self.t1 = nn.Parameter(sde.tvl1.weight_generator.weight)
        self.t2 = nn.Parameter(sde.tvl2.weight_generator.weight)
        self.t3 = nn.Parameter(sde.dfc1.weight)
        self.t3 = nn.Parameter(sde.dfc2.weight)

    def forward(self,t,x):
        return torch.tensor(v[0,np.argmin(abs(self.t_points-t))]/w[0])

rng = np.arange(0, 1.0, 0.01)
fcn = rng**3
#fcn = np.sin(2*np.pi*x)


ln = 100 #500 #training iterations
ls = torch.linspace(0, 1, ln)
#save = torch.empty(ys.size()[0],4,1)
save = torch.empty(ln,4,1)
ls = torch.empty(ln)
#print(k)
step_size = 10
#  ys = torchsde.sdeint(sde, y0, ts, method='euler')
y0 = 0.1
t_span = (0,1)
v = np.zeros((p.shape[0],p.shape[1]-1))
v_ = torch.zeros(p.shape[0])


def cub_loss(w1,w2,w3,w4):#,w2,w3,w4):
   # print(w1.dtype)
   # print(w1)
   # print(len(w1))
    sde.tvl1.weight_generator.weight = nn.Parameter(w1)
    #x = torch.tensor(w1)
    sde.tvl2.weight_generator.weight = nn.Parameter(w2)
    sde.dfc1.weight = nn.Parameter(w3)
    sde.dfc2.weight = nn.Parameter(w4)
    
    for n in range(p.shape[0]):  
        # print(n)
        om = lambda t: p[n,int(np.floor(t*dtt))]
        #  lambda t: p[n,int(np.floor(t*dtt))]
        time, vals = solve_stieltjes_ode(sde.f, sde.g, om , t_span, y0, p.shape[1])
         
        v[n,:] = w[n]*((time[:-1]**2 - vals[:-1])**2).detach().numpy()#.detach().numpy() # for specific loss function (t^2 - .)^2
        
        ## scale by weights
        #v[n,:] *= w[n]
    
     #   v_[n] = np.sum(v[n,:])*(1/p.shape[1]) #approximates E[\int f(t,X_t)dt]
         
        v_[n] = torch.sum(((time[:-1]**2 - vals[:-1])**2)*w[n])*(1/p.shape[1]) # retain computation graph
        
    return torch.sum(v_)


for k in range(ln):
    
    print(k)
    
   #g = grad(cub_loss)
    a = torch.tensor(np.array([[sde.tvl1.weight_generator.weight.clone().detach().numpy().item()]]),requires_grad=True)
    a.retain_grad()
    b = torch.tensor(np.array([[sde.tvl2.weight_generator.weight.clone().detach().numpy().item()]]),requires_grad=True)
    b.retain_grad()
    c = torch.tensor(np.array([[sde.dfc1.weight.clone().detach().numpy().item()]]),requires_grad=True)
    c.retain_grad()
    d = torch.tensor(np.array([[sde.dfc2.weight.clone().detach().numpy().item()]]),requires_grad=True)
    d.retain_grad()
    #grad = torch.autograd.grad(outputs = cub_loss(torch.tensor(np.array([[sde.tvl1.weight_generator.weight.clone().detach().numpy().item()]]),requires_grad=True)), inputs =[torch.tensor(np.array([[sde.tvl1.weight_generator.weight.clone().detach().numpy().item()]]),requires_grad=True)])
    
    cub_loss_plug = cub_loss(a,b,c,d)
    cub_loss_plug.retain_grad()
    #cub_loss_plug.retain_grad()
    cub_grad = cub_loss_plug.grad
    
    # def sum(a1,a2,a3,a4):
    # return (a1+a2+a3+a4).detach().item()
    
    #x = sum(a,b,c,d) 
    g = torch.autograd.grad(cub_loss_plug, [sde.tvl1.weight_generator.weight,sde.tvl2.weight_generator.weight,sde.dfc1.weight,sde.dfc2.weight], allow_unused=True) #
    #print(cub_grad)
    
    #t_points = torch.linspace(0, 1, p.shape[1])
    
    #y_final = ys[-1]
  #  target = np.zeros(ys.size())
    #target[:,1] = y
    ### More efficient way to do this assignment:
   # target[:] = fcn[:, None, None] 
    
    #loss = ((target - y_final) ** 2).sum(dim=1).mean(dim=0) 
    ls[k] = cub_loss_plug.clone().detach()
    
   # t0 = time.time()
    """
    torch.autograd.set_detect_anomaly(True)

    func = vectopath()

    odeint_adjoint(func, torch.tensor(y0), t_points[:-2])#,sde.tvl1.weight_generator.weight,sde.tvl2.weight_generator.weight,sde.dfc1.weight,sde.dfc2.weight)
    
    grad = torch.autograd.grad(cub_loss, inputs =[sde.tvl1.weight_generator.weight,sde.tvl2.weight_generator.weight,sde.dfc1.weight,sde.dfc2.weight])
   # g = grad(cub_loss, inputs =[sde.tvl1.weight_generator.weight,sde.tvl2.weight_generator.weight,sde.dfc1.weight,sde.dfc2.weight])
   # td = time.time() - t0 # time to compute gradient
    # ^ time_varying_linear.weight_generator -> mfc1 
    """
    
    g_mfc1 = g[0].detach().item()
    save[k][0] = sde.tvl1.weight_generator.weight.clone().detach()
    sde.tvl1.weight_generator.weight = nn.Parameter(sde.tvl1.weight_generator.weight.clone().detach() -  step_size*g_mfc1, requires_grad=True)  


    g_mfc2 = g[1].detach().item()
    save[k][1] = sde.tvl2.weight_generator.weight.clone().detach()
    sde.tvl2.weight_generator.weight = nn.Parameter(sde.tvl2.weight_generator.weight - step_size*g_mfc2, requires_grad=True) 
    
    
    g_dfc1 = g[2].detach().item()
    save[k][2] = sde.dfc1.weight.clone().detach()
    sde.dfc1.weight = nn.Parameter(sde.dfc1.weight.clone().detach() - step_size*g_dfc1, requires_grad=True)  
    
    
    g_dfc2 = g[3].detach().item()
    save[k][3] = sde.dfc2.weight.clone().detach()
    sde.dfc2.weight = nn.Parameter(sde.dfc2.weight.clone().detach() - step_size*g_dfc2, requires_grad=True)  
    
##############################
# Plot training optimization:
##############################

ti  = torch.linspace(0, 1, ln)
plot(ti, save, xlabel='$training iteration$', ylabel='$weight value$')
plt.plot(ti,ls)
plt.show()

###############################
# After training behavior:
###############################

batch_size, state_size, t_size = 5, 1, 100
#sde = SDE()
ts = torch.linspace(0, 1, t_size)
y0 = torch.full(size=(batch_size, state_size), fill_value=0.0)

with torch.no_grad():
    ys = torchsde.sdeint(sde, y0, ts, method='midpoint')  # (t_size, batch_size, state_size) = (100, 3, 1).


plt.plot(ts.tolist(), fcn)
plt.show()
plot(ts, ys, xlabel='$t$', ylabel='$Y_t$')


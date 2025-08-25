#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  6 11:12:05 2025

@author: lukesnow
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 25 16:42:20 2025

@author: lukesnow
"""

################
## Variational w Cubature
################

import torch
from torch import nn
import torchsde
import matplotlib.pyplot as plt
import torch.optim as optim
import numpy as np
import math
from numpy.linalg import svd
from torchdiffeq import odeint_adjoint as odeint
import time
import tracemalloc
import gc

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
       
       return t_points, y_values.unsqueeze(1)




### https://stackoverflow.com/questions/49852455/how-to-find-the-null-space-of-a-matrix-in-python-using-numpy
def nullspace(A, atol=1e-13, rtol=0):
    A = np.atleast_2d(A)
    u, s, vh = svd(A)
    tol = max(atol, rtol * s[0])
    nnz = (s >= tol).sum()
    ns = vh[nnz:].conj().T
    return ns




# %%

######################################################
# Order 5 Cubature Path Construction: Pre-Processing (without recom)
######################################################

disc_order = np.array([1,2,3,4,5,6,7])
path_constr = np.zeros(len(disc_order))
n_save = np.zeros(len(disc_order))

for do in range(len(disc_order)):
    
    start = time.time()
    
    #  print(cub_order[order])
    
    k = disc_order[do]
    
    t_d = np.zeros(k+1) # discretization time values
    g = 0.6 # gamma parameter
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
        #p = np.concatenate((p,p[:,0:dtn]),axis=1) 
        p = np.concatenate((p,np.zeros((p.shape[0],dtn))),axis=1) 
    
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
        thres = 5 # r+1
        
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
                   for h in range(len(find)):
                       if (h >= e):
                           find[h] = find[h] - 1
        
                w[find] = w_j
                j += 1
            """
      #  print(w[0])
      
    end = time.time()
    path_constr[do] = (end - start)
    
    x_vals = np.linspace(0,1,dtt)
        
    plt.figure(figsize=(8, 6))
    for n in range(p.shape[0]):
        plt.plot(x_vals, p[n,:])
    #plt.plot(x_vals, p[0,:])
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title(f'Cubature paths for dicretization level {k}')
    plt.grid(True)
    plt.show() 
    
    print(f'k: {do+1}, n: {p.shape[0]}')
    
    n_save[do] = p.shape[0]

plt.plot(disc_order, path_constr)
plt.xlabel("Cubature Discretization Order (k)")
plt.ylabel("Construction Time (sec)")
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.show()


plt.plot(disc_order, n_save)
plt.xlabel("Cubature Discretization Order (k)")
plt.ylabel("Total Paths (n)")
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.show()

######################################################
# ODE Cubature Solve
######################################################


# %%
##############################
# Model Init
##############################


# -- True data SDE (Stratonovich) --
class SDE_Data(nn.Module):
    noise_type = "diagonal"
    sde_type   = "stratonovich"
    def __init__(self):
        super().__init__()
    def f(self, t, x):
        B = x.size(0)
        drift = 1.0 + 0.5 * torch.sin(2 * torch.pi * t) #2.0 + 
     #   drift = torch.sin(2 * torch.pi * t) * x + 0.1 * torch.sin(x)
      #  drift = -x**3 + x + 0.1 * torch.sin(2 * torch.pi * t)
        return drift.expand(B,1)
    def g(self, t, x):
        B = x.size(0)
        diff = 0.1 + 0.05 * torch.cos(4 * torch.pi * t)
     #   diff = (0.2 + 0.1 * torch.sin(4 * torch.pi * t)) * (0.5 + 0.5 * torch.tanh(x))
        return diff.expand(B,1)

# -- Neural drift/diffusion for latent SDE --
class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, out_dim)
        )
    def forward(self, t, y):
        # y: either [B,D] or [T,B,D]; t: scalar, [B] or [T]
        if y.dim() == 2:
            # single time-step
            B, D = y.shape
            if t.dim() == 0:
                t2 = t.expand(B)
            else:
                t2 = t
            t2 = t2.unsqueeze(-1)                # [B,1]
            inp = torch.cat([t2, y], dim=-1)     # [B, D+1]
            return self.net(inp)                 # [B, out_dim]
        elif y.dim() == 3:
            # full trajectory
            T, B, D = y.shape
            if t.dim() == 1:
                t2 = t[:,None].expand(-1, B)     # [T,B]
            else:
                t2 = t.expand(T)               
                t2 = t2[:,None].expand(-1, B)
            t2 = t2.unsqueeze(-1)               # [T,B,1]
            inp = torch.cat([t2, y], dim=-1)     # [T,B,D+1]
            out = self.net(inp.view(-1, D+1))    # [(T*B), out_dim]
            return out.view(T, B, -1)           # [T,B,out_dim]
        else:
            raise ValueError(f"MLP.forward: unsupported y.dim()={y.dim()}")

class DiagDiff(nn.Module):
    def __init__(self, D):
        super().__init__()
        self.log_std = nn.Parameter(torch.zeros(D))
    def forward(self, t, y):
        # y: [B,D] or [T,B,D]
        std = torch.exp(self.log_std)           # [D]
        if y.dim() == 2:
            B, D = y.shape
            mat = torch.diag(std)               # [D,D]
            return mat.unsqueeze(0).expand(B, D, D)  # [B,D,D]
        elif y.dim() == 3:
            T, B, D = y.shape
            mat = torch.diag(std)               # [D,D]
            return mat.unsqueeze(0).unsqueeze(0).expand(T, B, D, D)  # [T,B,D,D]
        else:
            raise ValueError(f"DiagDiff.forward: unsupported y.dim()={y.dim()}")

class PosteriorSDE(torchsde.SDEIto):
    def __init__(self, f_phi, g_theta):
        super().__init__(noise_type="general")
        self._f, self._g = f_phi, g_theta
    def f(self, t, y): return self._f(t, y)
    def g(self, t, y): return self._g(t, y)


class LatentSDE(nn.Module):
    def __init__(self, D):
        super().__init__()
        self.f_theta = MLP(D+1, D)
        self.f_phi   = MLP(D+1, D)
        self.g_theta = DiagDiff(D)
        self.decoder = nn.Sequential(
            nn.Linear(D,64), nn.ReLU(),
            nn.Linear(64,1))
        
    def forward(self, z0, ts):
        # z0: [B,D], ts: [T]
        def fφ(t, y): return self.f_phi(t, y)
        def gθ(t, y): return self.g_theta(t, y)
        sde = PosteriorSDE(fφ, gθ)
        z = torchsde.sdeint(sde, z0, ts, method='euler')  # [T,B,D]
        return z

 # Define drift wrapper for batched cubature
class AugmentedDrift(nn.Module):
    def __init__(self, model, p_tensor, mult):
        super().__init__()
        self.model = model
        self.p_tensor = p_tensor  # [p, T, 1]
        self.mult = mult #multiplicative time scale factor
 
    def forward(self, t, y):  # y: [p, D]
        T_len = self.p_tensor.shape[1]*self.mult
        t_idx = min(int(t.item() * (T_len - 1)), T_len - 1)
        omega_t = self.p_tensor[:, int(t_idx/self.mult), 0]  # [p]
        f_phi_eval = self.model.f_phi(t, y)  # [p, D]
        g_theta_eval = self.model.g_theta(t, y)  # [p, D, D]
        #g_omega = torch.einsum("pdd,p->pd", g_theta_eval, omega_t)
       # g_omega = torch.bmm(g_theta_eval, omega_t.unsqueeze(-1)).squeeze(-1)  # [p, D]
        g_omega = (g_theta_eval.squeeze(-1) * omega_t.unsqueeze(-1))  # [p, D]
        return f_phi_eval + g_omega

    
# -- Plot helper --
def overlay_plot(ts, real, gen, xlabel, ylabel, title=''):
    ts = ts.cpu()
    real = real.squeeze().t().cpu()
    gen  = gen.squeeze().t().cpu()
    plt.figure()
    for i in range(min(real.shape[0], 10)):
        plt.plot(ts, real[i], color='blue',  alpha=0.5)
        plt.plot(ts, gen[i],  '--', color='orange', alpha=0.5)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend(['observed data','NSDE approximation'])
    plt.show()

    
MC = 10
#N_space = np.linspace(20,1000,5,dtype=int)
N_space = np.array([1])
#N_space = p.shape[1]*np.linspace(1,50,20,dtype=int)
train_itr = 100

# -- Storage --
loss_save = np.zeros((MC, train_itr))
loss_save_ = np.zeros((MC,train_itr))
loss_diff = np.zeros((MC, train_itr))

mem_pk_cub = np.zeros((MC,train_itr))
mem_pk_nsde = np.zeros((MC,train_itr))
tim_nsde = np.zeros(MC)
tim_cub = np.zeros(MC)

for mc in range(MC):
    
    batch_size = 50
    
    ##############################
    # SDE Training: MC
    ##############################
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    T, B, D = 100, 27, 1
  #  T =int(N_space[mc])
    ts = torch.linspace(0,1,T,device=device)
    dt = ts[1] - ts[0]
    
    # Generate true data once
    y0 = torch.full((B,1), 0.1, device=device)
    sde_data = SDE_Data().to(device)
    with torch.no_grad():
        y_true = torchsde.sdeint(sde_data, y0, ts, method='midpoint')  # [T,B,1]
        
    
    # Build model
    model = LatentSDE(D).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    sigma_obs = 0.1
    
    y0 = torch.full((27,1),0.1,device=device)
    z0 = y0.expand(-1, D)            # [B,D]
    with torch.no_grad():
        z = model(z0, ts)
        y_gen = model.decoder(z) 
       #! y_gen = z
    overlay_plot(ts, y_true, y_gen, xlabel='t', ylabel='x', title='MC pre-training')
    
    #batch_size = 500
    
    B_data = 27

    B_sim = 27
    
    ##:
    #T=98 #hard-coded for p.shape[1], so that y_true can be used for both NSDE and cubature evals
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    
    #model = LatentSDE(D).to(device)
    #^ build (initialize vfield parameters) once outside MC loop
    
    #model = LatentSDE(D).to(device)
    sigma_obs = 0.1
    loss_ = 0
    y0_sim = torch.full((B_sim, 1), 0.1, device=device)
    
    
    nsde_duration = 0
    
    #tracemalloc.start()
    
    for epoch in range(train_itr):
        
        if (epoch == int(train_itr/4)) or (epoch == int(train_itr/2)):
            y0 = torch.full((27,1),0.1,device=device)
            z0 = y0.expand(-1, D)            # [B,D]
            with torch.no_grad():
                z = model(z0, ts)
                y_gen = model.decoder(z) 
               #! y_gen = z
            overlay_plot(ts, y_true, y_gen, xlabel='t', ylabel='x', title='MC mid training')
            
        optimizer.zero_grad()
        loss_ = 0
        z0 = y0_sim.expand(-1, D)           # [B_sim, D]
        init_penalty = 0
        
        
        start = time.time()
        
        tracemalloc.start()
        
        z = model(z0, ts)  # [T, B_sim, D]
        
        
        fφ = model.f_phi(ts, z)
        fθ = model.f_theta(ts, z)
        gV = model.g_theta(ts, z)
        diag = gV.diagonal(dim1=-2, dim2=-1)
        u = (fφ - fθ) / diag
        kl = 0.5 * dt * (u.pow(2).sum() / B_sim)
        
        y_pred_all = model.decoder(z)  # [T, B_sim, 1]
        nll_ = 0
        
        for m in range(batch_size):
            idx = np.random.randint(0, B_data)
            y_target = y_true[:, idx:idx+1, :]  # [T,1,1]
            idc = np.random.choice(B_sim, 1)
            y_pred = y_pred_all[:, idc, :]      # [T,1,1]
            #nll = 0.5 * dt * ((y_pred - y_target)**2).sum() / sigma_obs**2
            #nll_ += nll
            nll_ += 0.5 * ((y_pred - y_target)**2).sum() / (ts.shape[0] * sigma_obs**2)
        
        nll_ /= batch_size
        
        λ = 10
        y0_pred = y_pred_all[0]
        y0_true = y_true[0]
        init_penalty = λ * ((y0_pred - y0_true)**2).sum()
        
        loss_ = kl + nll_ + init_penalty
       #loss_ /= batch_size
       
        current, peak = tracemalloc.get_traced_memory()
        mem_pk_nsde[mc,epoch] = peak
        tracemalloc.stop()
       
        end = time.time()
        nsde_duration += (end - start)
        
        print(f'training epoch: {epoch}')
        if epoch % 10 == 0:
           # print(f"[{epoch:03d}] Cubature loss={loss_.item():.4f}, kl={kl.item():.4f}, nll={nll.item():.4f}")
            print(f"[{mc:03d}, {epoch:03d}] NSDE loss={loss_.item():.4f}, kl={kl.item():.4f}, nll={nll_.item():.4f}") # Final overlay plot
        
        #loss = kl + nll + init_penalty
        loss_.backward()
        optimizer.step()
         
        loss_save_[mc, epoch] = loss_.item()
        
    
    print('\n')
    

    #current, peak = tracemalloc.get_traced_memory()
    #mem_pk_nsde[mc] = peak
    #tracemalloc.stop()

    
    nsde_duration /= train_itr
    tim_nsde[mc] = nsde_duration
    print(f'average duration: {nsde_duration}, peak memory: {peak}')
    
    y0 = torch.full((27,1),0.1,device=device)
    z0 = y0.expand(-1, D)            # [B,D]

    with torch.no_grad():
        z = model(z0, ts)
        y_gen = model.decoder(z) 
       #! y_gen = z
    overlay_plot(ts, y_true, y_gen, xlabel='t', ylabel='x', title='MC trained')


    
    # %%
    
    
    ##############################
    # SDE Training: Cubature
    ##############################
    
    from torchdiffeq import odeint_adjoint as odeint
    
    # -- Training --
    #def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    T, B, D = 100, p.shape[0], 1
    w_tensor = torch.tensor(w, dtype=torch.float32, device=device)
    # ts = torch.linspace(0,1,T,device=device)
    
    
    ts = torch.linspace(0,1,p.shape[1],device=device)
   # ts = torch.linspace(0,1,N_space[mc],device=device)
    dt = ts[1] - ts[0]
    
    # Generate true data once
    y0 = torch.full((B,1), 0.1, device=device)
    sde_data = SDE_Data().to(device)
    with torch.no_grad():
         y_true = torchsde.sdeint(sde_data, y0, ts, method='midpoint')  # [T,B,1]
    
     # Build model
    model = LatentSDE(D).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    sigma_obs = 0.1
    #loss_save_ = np.zeros(train_itr)
    #kl_save = np.zeros(train_itr)
    #nll_save = np.zeros(train_itr)
     
    y0 = torch.full((27,1),0.1,device=device)
    z0 = y0.expand(-1, D)            # [B,D]
    with torch.no_grad():
        z = model(z0, ts)
        y_gen = model.decoder(z) 
       #! y_gen = z
    overlay_plot(ts, y_true, y_gen, xlabel='t', ylabel='x', title='Cubature pre-training')


    # -- cubature solve parameters --
    t_span = (0,1)
    y_0 = y_true[0,0]

     
    #!:
    y0 = torch.full((1,1),0.1,device=device)
    
    #def f_augmented(t, y):
    #    omega_t = interpolate(ω_n, t)
    #    return model.f_phi(t, y) + model.g_theta(t, y) @ omega_t
    
    # Prepare tensors
    p_tensor = torch.tensor(p, dtype=torch.float32, device=device).unsqueeze(-1)  # [p, T, 1]
    w_tensor = torch.tensor(w, dtype=torch.float32, device=device)
    y0_batched = y_true[0, 0, 0].detach().expand(p.shape[0], 1)
    
    
    cub_duration = 0
    loss = 0
   # tracemalloc.start()
   # tracemalloc.reset_peak()
    
    for epoch in range(train_itr):
        
        if (epoch == int(train_itr/4)) or (epoch == int(train_itr/2)):
            y0 = torch.full((27,1),0.1,device=device)
            z0 = y0.expand(-1, D)            # [B,D]
            with torch.no_grad():
                z = model(z0, ts)
                y_gen = model.decoder(z) 
               #! y_gen = z
            overlay_plot(ts, y_true, y_gen, xlabel='t', ylabel='x', title='Cubature mid training')
            
            
        optimizer.zero_grad()
        
        start = time.time()
        tracemalloc.start()
       # tracemalloc.reset_peak()
        
        # ODE solve with adjoint
        drift_fn = AugmentedDrift(model, p_tensor, 1) #mult -> N_itr + 1, when evaluating vs. N
        
        #with torch.no_grad():
        v = odeint(drift_fn, y0_batched, ts, method='euler')  # [T, p, D]
        
        
        # KL divergence
        fφ = model.f_phi(ts, v)
        fθ = model.f_theta(ts, v)
        gV = model.g_theta(ts, v)
        diag = gV.diagonal(dim1=-2, dim2=-1)
        u = (fφ - fθ) / diag
        kl = 0.5 * (1 / p.shape[1]) * (u.pow(2).sum(0)[:, 0] * w_tensor).sum()
        
        # Decoder and NLL
        y_pred_all = model.decoder(v)  # [T, p, 1]
        nll_ = 0
        for _ in range(batch_size):
            idx = torch.randint(0, p.shape[0], (1,)).item()
            idc = torch.multinomial(w_tensor, 1).item()
            #with torch.no_grad():
            y_target = y_true[:, idx:idx + 1, :]
            y_pred = y_pred_all[:, idc:idc + 1, :]
            nll_ += 0.5 * ((y_pred - y_target)**2).sum() / (ts.shape[0] * sigma_obs**2)
        nll_ /= batch_size
        
        # Initial penalty
        λ = 10
        y0_pred = y_pred_all[0]
        y0_true = y_true[0]
        init_penalty = λ * ((y0_pred - y0_true)**2).sum()
        
        # Final loss
        loss = kl + nll_ + init_penalty
        
        current, peak = tracemalloc.get_traced_memory()
        mem_pk_cub[mc,epoch] = peak
        tracemalloc.clear_traces()
        tracemalloc.stop()
        
        end=time.time()
        cub_duration += (end - start)
        
        loss.backward()
        optimizer.step()
    
        loss_save[mc,epoch] = loss.item()
    
        if epoch % 10 == 0:
            print(f"[{mc:03d}, {epoch:03d}] Cubature loss={loss.item():.4f}, kl={kl.item():.4f}, nll={nll_.item():.4f}")
            #print(f"[{epoch:03d}] NSDE loss={loss_.item():.4f}, kl_={kl_.item():.4f}, nll_={nll_.item():.4f}") # Final overlay plot
            
      #  del model, optimizer
      #  del v, fφ, fθ, gV, diag, u, y_pred_all, drift_fn
      #  torch.cuda.empty_cache()
      #  gc.collect()
    
    #current, peak = tracemalloc.get_traced_memory()
    #mem_pk_cub[mc] = peak
    #tracemalloc.clear_traces()
    #tracemalloc.stop()
    
    cub_duration /= train_itr
    tim_cub[mc] = cub_duration


    #################
    y0 = torch.full((27,1),0.1,device=device)
    z0 = y0.expand(-1, D)            # [B,D]
    with torch.no_grad():
        z = model(z0, ts)
        y_gen = model.decoder(z) 
       #! y_gen = z
    overlay_plot(ts, y_true, y_gen, xlabel='t', ylabel='x', title='Cubature trained')

### plot loss vs training itr:
plt.plot(np.linspace(0,train_itr,train_itr),np.mean(loss_save,0),'b-')
plt.plot(np.linspace(0,train_itr,train_itr),np.mean(loss_save_,0),'r-')
#plt.plot(np.linspace(0,train_itr,train_itr),loss_save_,'r-')
plt.xlabel("Training Iteration")
plt.ylabel("Variational Loss")
plt.legend(['Cubature evaluation','Monte-Carlo evaluation'])
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.show()


### plot loss vs compute time:
# cubature loss plot:
cub_ax = np.linspace(0,train_itr,train_itr) * np.mean(tim_cub)
plt.plot(cub_ax, np.mean(loss_save,0),'b-')
# NSDE loss plot:
nsde_ax = np.linspace(0,train_itr,train_itr) * np.mean(tim_nsde)
plt.plot(np.linspace(0,train_itr,train_itr),np.mean(loss_save_,0),'r-')
plt.xlabel("Compute time (sec)")
plt.ylabel("Variational Loss")
plt.legend(['Cubature evaluation','Monte-Carlo evaluation'])
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.xscale('log')
plt.show()

### plot avg memory reuirement vs training iteration
plt.plot(np.linspace(0,train_itr,train_itr),np.mean(mem_pk_cub,0)/(10**6),'b-')
plt.plot(np.linspace(0,train_itr,train_itr),np.mean(mem_pk_nsde,0)/(10**6),'r-')
#plt.plot(np.linspace(0,train_itr,train_itr),loss_save_,'r-')
plt.xlabel("Training Iteration")
plt.ylabel("Memory Requirement (MB)")
plt.legend(['Cubature evaluation','Monte-Carlo evaluation'])
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.yscale('log')
plt.show()



print(f'peak memory average: NSDE: {np.mean(mem_pk_nsde)}, Cubature: {np.mean(mem_pk_cub)}')
print(f'average loss evaluation time: NSDE: {np.mean(tim_nsde)}, Cubature: {np.mean(tim_cub)}')

#print(f'average duration: {cub_duration}, peak memory: {peak}')
"""
rng = np.linspace(1,len(N_space),len(N_space))
rng = N_space

plt.plot(rng,mem_pk_nsde,'r-')
plt.plot(rng,mem_pk_cub,'b-')
plt.yscale('log')
plt.xscale('log')
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.show()

plt.plot(rng,tim_nsde,'r-')
plt.plot(rng,tim_cub,'b-')
plt.yscale('log')
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.show()
"""

#if __name__ == '__main__':
#    main()
    
    



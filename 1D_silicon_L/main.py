import torch
import numpy as np
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset,RandomSampler
import time
import array
from pathlib import Path
import sys
import os
import matplotlib
from matplotlib import pyplot as plt
from mesh_gen import *
from beta_train import beta_train
from bte_train import bte_train, bte_test

'''
	PINN for 1D cross-plane phonon transport in silicon.
	Here the system length is included as an input variable to enable parametric learning.
	In this case reference temperature Tref = 300K, while the boundary temperature Th = 301K, Tc = 299K
	The boundary temperature difference can be adjusted for your own problem.
	Scaling factor Beta mentioned in the paper is trained via "beta_train".
'''
epochs = 30000
path = "./train/"
Nl = 9  # number of length samples
logL = np.linspace(0,4,Nl).reshape(-1,1) # log10(L) ranges from 10 nm to 100 micron 
index = 1

Nx = 40  # number of spatial points
Nk = 10  # number of frequency bands
Ns = 16  # number of quadrature points
Np = 3   # number of phonon branches
Tref = 300   # reference temperature
Nt = 20  # number of temperature intervals for beta training
dT = 1  # boudanry temperature difference from Tref, could be adjusted

x,mu,w,k = OneD_mesh(Nx,Ns,Nk) # sample points in the input domain

beta_in = np.linspace(-1,1,Nt).reshape(-1,1)  # for beta training: input range [-dT, dT] scaled by dT

#===============================================================
#=== model training
#===============================================================

learning_rate = 5e-3
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# first train beta mentioned in the paper
beta_train(k,beta_in,Nk,Nt,dT,Tref,learning_rate,epochs,path,device)

# train the model for BTE solution
bte_train(x,mu,w,k,Nx,Ns,Nk,Np,logL,Tref,dT,learning_rate,epochs,path,device)

#===============================================================
#=== model testing
#===============================================================

Nx = 80
Ns = 64  # number of quadrature points
Nl = 17
x = np.linspace(0,1,Nx+2)[1:Nx+1].reshape(-1,1)
logL = np.linspace(0,4,Nl).reshape(-1,1)
_,mu,w,_ = OneD_mesh(Nx,Ns,Nk)

bte_test(x,mu,w,k,Nx,Ns,Nk,Np,logL,Tref,dT,index,path,device)

#===============================================================
#=== results ploting
#===============================================================

Data = np.load(str(index)+'NG_1d_L.npz')
x = Data['x']
T = Data['T2']
q = Data['q'].mean(0)
ind1 = np.arange(0,Nl,4)
ind2 = np.arange(0,Nl,2)

plt.figure(figsize=(12, 5))
plt.subplot(1,2,1)
plt.plot(x,T[:,ind1],'r--',linewidth=2.5)
plt.ylabel(r'$T^{*}$')
plt.xlabel(r'$X$')
plt.axis([0,1,0,1])

plt.subplot(1,2,2)
plt.plot(10**(logL[ind2]-8),q[ind2],'ro')
plt.xscale("log")
plt.ylabel(r'$k_{eff}/k_{bulk}$')
plt.xlabel(r'$L$ (m)')
plt.savefig('res_1d.png', dpi=300, bbox_inches='tight')






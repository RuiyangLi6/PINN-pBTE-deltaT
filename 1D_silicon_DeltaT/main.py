import torch
import numpy as np
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader,TensorDataset,RandomSampler
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
	PINN for 1D cross-plane phonon transport with temperature difference as input.
	In this case temperature is within the range [200K, 400K]
	Scaling factor Beta mentioned in the paper is trained via "beta_train".
'''
epochs = 30000
path = "./train/"
index = 1
Nl = 9   # number of length samples
L = 1e4  # L = 1 => 10 nm; L = 1e4 => 100 micron

Tref = 300   # reference temperature
Ts = 100 # boudary temperature difference from the reference temperature, could be adjusted
dT = np.array([[10,30,50,75,100]]).T/Ts  # input boudanry temperature difference scaled by Ts, we train for 5 cases here

Nx = 40  # number of spatial points
Nk = 10  # number of frequency bands
Ns = 16  # number of quadrature points
Np = 3   # number of phonon branches
Nt = 20   # number of temperature intervals for beta training
beta_in = np.linspace(-1,1,Nt).reshape(-1,1)

x,mu,w,k = OneD_mesh(Nx,Ns,Nk) # sample points in the input domain

#===============================================================
#=== model training
#===============================================================

learning_rate = 5e-3
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# first train beta mentioned in the paper
beta_train(k,beta_in,Nk,Nt,Ts,Tref,learning_rate,epochs,path,device)

# train the model for BTE solution
bte_train(x,mu,w,k,dT,Nx,Ns,Nk,Np,L,Tref,Ts,learning_rate,epochs,path,device)

#===============================================================
#=== model testing
#===============================================================

Nx = 80
Ns = 64
x = np.linspace(0,1,Nx+2)[1:Nx+1].reshape(-1,1)
_,mu,w,_ = OneD_mesh(Nx,Ns,Nk)

bte_test(x,mu,w,k,Nx,Ns,Nk,Np,L,Tref,Ts,dT,index,path,device)

#===============================================================
#=== results ploting
#===============================================================

Data = np.load(str(index)+'NG_1d_T.npz')
x = Data['x']
T = Data['T2'] + Tref

Tc = Tref - np.squeeze(dT*Ts)
Th = Tref + np.squeeze(dT*Ts)
gamma = -1.5698262734617936 # parameter for analytical solution under this temperature gradient
Ta = (x*Tc**(1+gamma) + (1-x)*Th**(1+gamma))**(1/(gamma+1)) # analytical solution in the diffusive limit

plt.figure(figsize=(6, 5))
line1 = plt.plot(x,Ta,'k-',linewidth=2.5)
line2 = plt.plot(x,T,'r--',linewidth=2.5)
plt.ylabel(r'$T$(K)')
plt.xlabel(r'$X$')
plt.legend([line1[0],line2[0]],['Fourier','PINN'],frameon=False, loc='lower left')
plt.savefig('res_1d.png', dpi=300, bbox_inches='tight')






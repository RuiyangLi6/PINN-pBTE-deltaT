import torch
import numpy as np
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader,TensorDataset,RandomSampler
import time
import sys
from mesh_gen import *
from model import Net

def beta_train(k,dT,Nk,Nt,Ts,Tref,learning_rate,epochs,path,device):
	net = Net(3, 1, 30).to(device)
	optimizer = optim.Adam(net.parameters(), lr=learning_rate, betas=(0.9,0.99), eps=1e-10)

	def init_normal(m):
		if type(m) == nn.Linear:
			nn.init.kaiming_normal_(m.weight)

	net.apply(init_normal)
	net.train()

	############################################################################

	def criterion(k,p,dT):
		k.requires_grad = True
		p.requires_grad = True
		dT.requires_grad = True

		scale = net(torch.cat((k,p,dT),1))*dfdT
		loss_1 = (scale*dT*Ts)/df - 1

		##############
		# MSE LOSS
		loss_f = nn.MSELoss()

		loss1 = loss_f(loss_1,torch.zeros_like(loss_1))

		return loss1

	###################################################################

	# Main loop
	Loss_min = 100
	Loss_list = []
	tic = time.time()

	k = np.tile(k,(2*Nt,1))
	p = np.vstack((np.zeros((Nt*Nk,1)),np.ones((Nt*Nk,1))))
	om,_,_,_,dfdT = param_phonon(k,p,Tref)

	om = torch.FloatTensor(om).to(device)
	dfdT = torch.FloatTensor(dfdT).to(device)

	dT = torch.FloatTensor(dT).repeat(1,Nk).reshape(-1,1).repeat(2,1).to(device)
	p = torch.FloatTensor(p).to(device)
	k = torch.FloatTensor(k).to(device)
	k = k/(np.pi*2/a)

	fEq = 1/(torch.exp(hkB*om/Tref)-1)
	ff = 1/(torch.exp(hkB*om/(Tref+dT*Ts))-1)
	df = ff - fEq

	for epoch in range(epochs):
		net.zero_grad()
		loss = criterion(k,p,dT)
		loss.backward()
		optimizer.step()
		Loss = loss.item()
		Loss_list.append(loss.item())
		if epoch%2000 == 0:
			print('Train Epoch: {}  Loss: {:.6f}'.format(epoch,loss.item()))
		if Loss < Loss_min:
			torch.save(net.state_dict(),path+"model_beta.pt")
			Loss_min = Loss

	toc = time.time()
	elapseTime = toc - tic
	print ("elapse time in parallel = ", elapseTime)
	np.savetxt(path+'Loss_beta.txt',np.array(Loss_list), fmt='%.6f')

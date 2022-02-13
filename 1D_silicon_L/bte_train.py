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

def bte_train(x,mu,w,k,Nx,Ns,Nk,Np,L,Tref,dT,learning_rate,epochs,path,device):
	net0 = Net(5, 8, 30).to(device)
	net1 = Net(2, 5, 30).to(device)

	optimizer0 = optim.Adam(net0.parameters(), lr=learning_rate, betas=(0.9,0.99), eps=1e-10)
	optimizer1 = optim.Adam(net1.parameters(), lr=learning_rate, betas=(0.9,0.99), eps=1e-10)

	def init_normal(m):
		if type(m) == nn.Linear:
			nn.init.kaiming_normal_(m.weight)

	net0.apply(init_normal)
	net1.apply(init_normal)
	net0.train()
	net1.train()

	net2 = Net(3, 1, 30).to(device)
	net2.load_state_dict(torch.load(path+"model_beta.pt",map_location=device))
	net2.eval()

	############################################################################

	def criterion(x,mu,w,mu0,mu1,k,kb,L):
		x.requires_grad = True

		######### Interior points ##########
		T = net1(torch.cat((x,L),1))*dT
		scale0 = net2(torch.cat((k,torch.zeros_like(k),T/dT),1)) # scaling factor (beta in the paper)
		scale1 = net2(torch.cat((k,torch.ones_like(k),T/dT),1))
		scale = torch.cat((scale0.reshape(-1,Ns)[:,0].reshape(-1,1).reshape(-1,Nk),scale0.reshape(-1,Ns)[:,0].reshape(-1,1).reshape(-1,Nk),scale1.reshape(-1,Ns)[:,0].reshape(-1,1).reshape(-1,Nk)),1).reshape(-1,1)
		tau0 = param_tau(k*np.pi*2/a,torch.zeros_like(k),T+Tref) 
		tau1 = param_tau(k*np.pi*2/a,torch.ones_like(k),T+Tref)
		taus = torch.cat((tau0.reshape(-1,Ns)[:,0].reshape(-1,1).reshape(-1,Nk),tau0.reshape(-1,Ns)[:,0].reshape(-1,1).reshape(-1,Nk),tau1.reshape(-1,Ns)[:,0].reshape(-1,1).reshape(-1,Nk)),1).reshape(-1,1)

		f0_in = torch.cat((x,mu,k,torch.zeros_like(x),L),1)
		f1_in = torch.cat((x,mu,k,torch.ones_like(x),L),1)
		f0 = net0(f0_in)/(10**L)*dT
		f1 = net0(f1_in)/(10**L)*dT

		f0_x = torch.autograd.grad(f0+scale0*T,x,grad_outputs=torch.ones_like(x).to(device),create_graph=True)[0]
		f1_x = torch.autograd.grad(f1+scale1*T,x,grad_outputs=torch.ones_like(x).to(device),create_graph=True)[0]

		f_x = torch.cat((f0_x.reshape(-1,Ns*Nk),f0_x.reshape(-1,Ns*Nk),f1_x.reshape(-1,Ns*Nk)),1).reshape(-1,1)
		f = torch.cat(((f0+scale0*T).reshape(-1,Ns*Nk),(f0+scale0*T).reshape(-1,Ns*Nk),(f1+scale1*T).reshape(-1,Ns*Nk)),1).reshape(-1,1)
		sum_fx = torch.matmul(f_x.reshape(-1,Ns),w*mu[0:Ns].reshape(-1,1)).reshape(-1,1)
		sum_f = torch.matmul(f.reshape(-1,Ns), w).reshape(-1,1).to(device)
		dq = torch.matmul(sum_fx.reshape(-1,Nk*Np),hbar*om*D*dfdT*wk*vk*vk).reshape(-1,1)
		T1 = torch.matmul((sum_f/taus).reshape(-1,Nk*Np),om*D*dfdT*vk/(4*np.pi))/torch.matmul((scale/taus).reshape(-1,Nk*Np),om*D*dfdT*vk)
		T1 = T1.reshape(-1,1).repeat(1,Nk*Ns).reshape(-1,1)

		######### Isothermal boundary ##########
		Tc = net1(torch.cat((torch.ones_like(Lb),Lb),1))*dT
		Th = net1(torch.cat((torch.zeros_like(Lb),Lb),1))*dT

		sc = net2(torch.cat((kb,pb,Tc/dT),1))
		sh = net2(torch.cat((kb,pb,Th/dT),1))

		c_in = torch.cat((torch.ones_like(kb),mu1,kb,pb,Lb),1)
		fc = net0(c_in)/(10**Lb)*dT + sc*Tc

		h_in = torch.cat((torch.zeros_like(kb),mu0,kb,pb,Lb),1)
		fh = net0(h_in)/(10**Lb)*dT + sh*Th

		######### Loss ##########
		loss_1 = (mu*f0_x + f0/(v0*tau0)*(10**L))/dT # bte residual for branch 0
		loss_2 = (mu*f1_x + f1/(v1*tau1)*(10**L))/dT # bte residual for branch 1
		loss_3 = (T - T1)
		loss_4 = dq/TC
		loss_5 = (fc - cb)/dT
		loss_6 = (fh - hb)/dT

		##############
		# MSE LOSS
		loss_f = nn.MSELoss()

		loss1 = loss_f(loss_1,torch.zeros_like(loss_1))
		loss2 = loss_f(loss_2,torch.zeros_like(loss_2))
		loss3 = loss_f(loss_3,torch.zeros_like(loss_3))
		loss4 = loss_f(loss_4,torch.zeros_like(loss_4))
		loss5 = loss_f(loss_5,torch.zeros_like(loss_5))
		loss6 = loss_f(loss_6,torch.zeros_like(loss_6))

		return loss1,loss2,loss3,loss4,loss5,loss6

	###################################################################

	# Main loop
	Loss_min = 100
	Loss_list = []
	tic = time.time()

	p = np.vstack((np.zeros_like(k),np.zeros_like(k),np.ones_like(k)))
	om,vk,D,tau,dfdT = param_phonon(np.tile(k,(Np,1)),p,Tref)

	Nl = len(L)
	wk = np.pi*2/a/Nk
	TC = np.zeros((Nl,1))
	for i in range(Nl):
		TC[i] = (1/3)*np.sum(hbar*om*D*dfdT*vk**3*tau*wk)*dT*2*1e11/(10**L[i])*4*np.pi
	TC = torch.FloatTensor(TC).repeat(Nx,1).to(device)

	om = torch.FloatTensor(om).to(device)
	vk = torch.FloatTensor(vk).to(device)
	D = torch.FloatTensor(D).to(device)
	dfdT = torch.FloatTensor(dfdT).to(device)
	tau = torch.FloatTensor(tau).to(device)

	x = torch.FloatTensor(x).repeat(1,Ns*Nk*Nl).reshape(-1,1).to(device)
	mu0 = torch.FloatTensor(mu[mu>0].reshape(-1,1)).repeat(Nk*Nl*2,1).to(device)
	mu1 = torch.FloatTensor(mu[mu<0].reshape(-1,1)).repeat(Nk*Nl*2,1).to(device)
	mu = torch.FloatTensor(mu).repeat(Nx*Nk*Nl,1).to(device)
	w = torch.FloatTensor(w).to(device)

	Lb = torch.FloatTensor(L).repeat(1,int(Ns/2)*Nk).reshape(-1,1).repeat(2,1).to(device)
	L = torch.FloatTensor(L).repeat(1,Ns*Nk).reshape(-1,1).repeat(Nx,1).to(device)	
	kb = torch.FloatTensor(k).repeat(1,int(Ns/2)).reshape(-1,1).repeat(Nl,1).to(device)
	k = torch.FloatTensor(k).repeat(1,Ns).reshape(-1,1).repeat(Nx*Nl,1).to(device)
	v0,v1 = group_velocity(k)
	k = k/(np.pi*2/a)
	pb = torch.cat((torch.zeros_like(kb),torch.ones_like(kb)),0)
	kb = kb.repeat(2,1)/(np.pi*2/a)
	hb = net2(torch.cat((kb,pb,torch.ones_like(kb)),1)).detach()*dT
	cb = net2(torch.cat((kb,pb,-torch.ones_like(kb)),1)).detach()*(-dT)

	for epoch in range(epochs):
		net0.zero_grad()
		net1.zero_grad()
		loss1,loss2,loss3,loss4,loss5,loss6 = criterion(x,mu,w,mu0,mu1,k,kb,L)
		loss = loss1 + loss2 + loss3 + loss4 + loss5 + loss6
		loss.backward()
		optimizer0.step()
		optimizer1.step()
		Loss = loss.item()
		Loss_list.append([loss1.item(),loss2.item(),loss3.item(),loss4.item(),loss5.item(),loss6.item()])
		if epoch%2000 == 0:
			print('Train Epoch: {}  Loss: {:.6f}  {:.6f}  {:.6f}  {:.6f}  {:.6f}'.format(epoch,loss1.item(),loss3.item(),loss4.item(),loss5.item(),loss6.item()))
			torch.save(net1.state_dict(),path+"train_ng_epoch"+str(epoch)+"f.pt")
		if Loss < Loss_min:
			torch.save(net0.state_dict(),path+"model0.pt")
			torch.save(net1.state_dict(),path+"model1.pt")
			Loss_min = Loss

	toc = time.time()
	elapseTime = toc - tic
	print ("elapse time in parallel = ", elapseTime)
	np.savetxt(path+'Loss.txt',np.array(Loss_list), fmt='%.6f')


def bte_test(x,mu,w,k,Nx,Ns,Nk,Np,L,Tref,dT,index,path,device):
	net0 = Net(5, 8, 30).to(device)
	net0.load_state_dict(torch.load(path+"model0.pt",map_location=device))
	net0.eval()

	net1 = Net(2, 5, 30).to(device)
	net1.load_state_dict(torch.load(path+"model1.pt",map_location=device))
	net1.eval()

	net2 = Net(3, 1, 30).to(device)
	net2.load_state_dict(torch.load(path+"model_beta.pt",map_location=device))
	net2.eval()

	################################################################
	p = np.vstack((np.zeros_like(k),np.zeros_like(k),np.ones_like(k)))
	kp = np.tile(k,(Np,1))
	om,vk,D,tau,dfdT = param_phonon(kp,p,Tref)
	wk = np.pi*2/a/Nk
	om = torch.FloatTensor(om).to(device)
	vk = torch.FloatTensor(vk).to(device)
	D = torch.FloatTensor(D).to(device)
	tau = torch.FloatTensor(tau).to(device)
	dfdT = torch.FloatTensor(dfdT).to(device)

	mu = torch.FloatTensor(mu).repeat(Nx*Nk,1).to(device)
	w = torch.FloatTensor(w).to(device)
	k = torch.FloatTensor(k).repeat(1,Ns).reshape(-1,1).repeat(Nx,1).to(device)
	k = k/(np.pi*2/a)

	T1 = np.zeros((Nx,len(L))) # two ways to calculate pseudo-temperautre
	T2 = np.zeros((Nx,len(L)))
	q = np.zeros((Nx,len(L)))
	tic = time.time()
	x1 = torch.FloatTensor(x).repeat(1,Ns*Nk).reshape(-1,1).to(device)
	TC = ((1/3)*torch.sum(hbar*om*D*dfdT*vk**3*tau*wk)*dT*2*1e11*4*np.pi)

	for j in range(len(L)):
		L1 = torch.FloatTensor(L[j]).repeat(Ns*Nk*Nx,1).to(device)

		T = net1(torch.cat((x1,L1),1))*dT
		scale0 = net2(torch.cat((k,torch.zeros_like(k),T/dT),1))
		scale1 = net2(torch.cat((k,torch.ones_like(k),T/dT),1))
		scale = torch.cat((scale0.reshape(-1,Ns)[:,0].reshape(-1,1).reshape(-1,Nk),scale0.reshape(-1,Ns)[:,0].reshape(-1,1).reshape(-1,Nk),scale1.reshape(-1,Ns)[:,0].reshape(-1,1).reshape(-1,Nk)),1).reshape(-1,1)
		tau0 = param_tau(k*np.pi*2/a,torch.zeros_like(k),T+Tref) 
		tau1 = param_tau(k*np.pi*2/a,torch.ones_like(k),T+Tref)
		taus = torch.cat((tau0.reshape(-1,Ns)[:,0].reshape(-1,1).reshape(-1,Nk),tau0.reshape(-1,Ns)[:,0].reshape(-1,1).reshape(-1,Nk),tau1.reshape(-1,Ns)[:,0].reshape(-1,1).reshape(-1,Nk)),1).reshape(-1,1)

		f0_in = torch.cat((x1,mu,k,torch.zeros_like(x1),L1),1)
		f1_in = torch.cat((x1,mu,k,torch.ones_like(x1),L1),1)
		f0 = net0(f0_in)/(10**L1)*dT + scale0*T
		f1 = net0(f1_in)/(10**L1)*dT + scale1*T
		f = torch.cat((f0.reshape(-1,Ns*Nk),f0.reshape(-1,Ns*Nk),f1.reshape(-1,Ns*Nk)),1).reshape(-1,1)

		sum_f = torch.matmul(f.reshape(-1,Ns), w).reshape(-1,1)
		sum_fx = torch.matmul(f.reshape(-1,Ns), w*mu[0:Ns].reshape(-1,1)).reshape(-1,1)
		deltaT = torch.matmul((sum_f/taus).reshape(-1,Nk*Np),om*D*dfdT*vk/(4*np.pi))/torch.matmul((scale/taus).reshape(-1,Nk*Np),om*D*dfdT*vk)
		deltaT = deltaT.reshape(-1,1)
		q[:,j] = np.squeeze(torch.matmul(sum_fx.reshape(-1,Nk*Np), hbar*om*D*dfdT*wk*vk*vk).reshape(-1,1).cpu().data.numpy()/TC.cpu().data.numpy())*(10**L[j])
		T1[:,j] = T.reshape(-1,Ns*Nk)[:,0].cpu().data.numpy()
		T2[:,j] = np.squeeze(deltaT.cpu().data.numpy())

	toc = time.time()
	elapseTime = toc - tic
	print ("elapse time = ", elapseTime)
	np.savez(str(int(index))+'NG_1d_L',x = x,T1 = T1/dT/2 + 0.5,T2 = T2/dT/2 + 0.5,q = q)

import torch
import numpy as np
import torch.nn as nn

class Net(nn.Module):
	def __init__(self, input_n, NL, NN):
		super(Net, self).__init__()
		self.input_layer = nn.Linear(input_n, NN)
		self.hidden_layers = nn.ModuleList([nn.Linear(NN, NN) for i in range(NL-1)])
		self.output_layer = nn.Linear(NN, 1)

	def forward(self, x):
		o = self.act(self.input_layer(x))
		for i,li in enumerate(self.hidden_layers):
			o = self.act(li(o))
		output = self.output_layer(o)

		return  output

	def act(self, x):
		return x * torch.sigmoid(x)

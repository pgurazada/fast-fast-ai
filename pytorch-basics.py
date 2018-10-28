import torch
import torch.nn as nn
import torch.nn.functional as F 
import torchvision

# We are transcrining the talk into a script. The outout checks are done using
# print statements

# What you expect from numpy works also for torch

torch.eye(3)

Y = torch.rand((5, 3))

# print(Y.shape)

# print(torch.inverse(Y.t() @ Y))

# Torch has an autgrad as expected; computation of gradients is turned off by default 

x = torch.ones(1)
w = torch.ones(1) * 2
# print(w.requires_grad)

total = w * x

# total.backward() # throws up an error since we havent explicitly asked for gradients

x.requires_grad_(True)
w.requires_grad_(True)

total = w * x

total.backward() # No more an error

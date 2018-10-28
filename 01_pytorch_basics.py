
# coding: utf-8

# # PyTorch Basics

# ## Init, helpers, utils, ...

# In[1]:


import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


# In[2]:


import matplotlib.pyplot as plt
from pprint import pprint
import numpy as np

from IPython.core.debugger import set_trace

get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


from ppt.utils import attr


# # Tensors
# tensors - the atoms of machine learning

# ## Tensors in numpy and pytorch

# In[4]:


import numpy as np
from numpy.linalg import inv
from numpy.linalg import multi_dot as mdot


# In[5]:


import torch


# In[6]:


# numpy
np.eye(3)


# In[7]:


# torch
torch.eye(3)


# In[8]:


# numpy
X = np.random.random((5, 3))
X


# In[9]:


# pytorch
Y = torch.rand((5, 3))
Y


# In[10]:


X.shape


# In[11]:


Y.shape


# In[12]:


# numpy
X.T @ X


# In[13]:


# torch
Y.t() @ Y


# In[14]:


# numpy
inv(X.T @ X)


# In[15]:


# torch
torch.inverse(Y.t() @ Y)


# ## More on PyTorch Tensors

# Operations are also available as methods.

# In[16]:


A = torch.eye(3)
A.add(1)


# In[17]:


A


# Any operation that mutates a tensor in-place is post-fixed with an _. For example: x.copy_(y), x.t_(), will change x.

# In[18]:


A.add_(1)
A


# ## Indexing and broadcasting
# It works as expected:

# In[22]:


A


# In[19]:


A[0, 0]


# In[23]:


A[0]


# In[21]:


A[0:2]


# In[24]:


A[:, 1:3]


# ## Converting

# In[25]:


A = torch.eye(3)
A


# In[26]:


# torch --> numpy
A.numpy()


# In[27]:


# numpy --> torch
torch.from_numpy(np.eye(3))


# # Autograd
# Prior to `v0.4` PyTorch used the class `Variable` to record gradients. You had to wrap `Tensor`s in `Variable`s.
# `Variable`s behaved like `Tensors`.
# 
# With `v0.4` `Tensor` can record gradients directly if you tell it do do so, e.g. `torch.ones(3, requires_grad=True)`.
# There is no need for `Variable` anymore.
# 
# Ref:
# - https://pytorch.org/docs/stable/autograd.html
# - https://pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html

# In[28]:


from torch import autograd  # you rarely use it directly


# In[29]:


w = torch.ones(1)
w.requires_grad


# In[30]:


z = torch.ones(1) * 2
z.requires_grad


# In[31]:


total = w + z
total


# In[32]:


# What is going to happen here?
total.backward()


# In[33]:


w = torch.ones(1, requires_grad=True)
w.requires_grad


# In[34]:


total = w + z
total.requires_grad


# In[35]:


total.backward()


# In[36]:


w.grad


# In[37]:


with torch.no_grad():
    total = w + z

total.requires_grad


# # But what about the GPU?
# How do I use the GPU?
# 
# If you have a GPU make sure that the right pytorch is installed
# 
# ```
# conda install pytorch torchvision cuda91 -c pytorch
# ```
# Check https://pytorch.org/ for details.

# In[38]:


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device


# If you have a GPU you should get something like: 
# `device(type='cuda', index=0)`
# 
# You can move data to the GPU by doing `.to(device)`.

# In[39]:


data = torch.eye(3)
data.to(device)


# Note: before `v0.4` one had to use `.cuda()` and `.cpu()` to move stuff to and from the GPU.
# This littered the code with many:
# ```python
# if CUDA:
#     model = model.cuda()
# ```

# # LinReg with PyTorch, Gradient Descent, and GPU

# In[ ]:


from sklearn.datasets import make_regression

n_features = 1
n_samples = 100

X, y = make_regression(
    n_samples=n_samples,
    n_features=n_features,
    noise=10,
)

fix, ax = plt.subplots()
ax.plot(X, y, ".")


# In[ ]:


X = torch.from_numpy(X).float()
y = torch.from_numpy(y.reshape((n_samples, n_features))).float()


# In[ ]:


from torch import nn
from torch import optim


class LinReg(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.beta = nn.Linear(input_dim, 1)
        
    def forward(self, X):
        return self.beta(X)


# In[ ]:


# Move everything to GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = LinReg(n_features).to(device)  # <-- here
optimizer = optim.SGD(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

X, y = X.to(device), y.to(device)  # <-- here


# In[ ]:


# Train step
model.train()
optimizer.zero_grad()

y_ = model(X)
loss = criterion(y_, y)

loss.backward()
optimizer.step()

# Eval
model.eval()
with torch.no_grad():
    y_ = model(X)    

# Vis
fig, ax = plt.subplots()
ax.plot(X.cpu().numpy(), y_.cpu().numpy(), ".", label="pred")
ax.plot(X.cpu().numpy(), y.cpu().numpy(), ".", label="data")
ax.set_title(f"MSE: {loss.item():0.1f}")
ax.legend();


# # Debugging
# 
# **Q: "No debugger for your code. What do you think?"**
# 
# **A: "I would NOT be able to code!"**
# 
# - Who does "print-line-debugging"?
# - Who likes debugging in tensorflow?
# - What is the intersection of those two groups?
# 
# 
# ## IPDB cheatsheet
# IPython Debugger
# 
# Taken from http://frid.github.io/blog/2014/06/05/python-ipdb-cheatsheet/
# 
# - h(help): Print help
# 
# - n(ext): Continue execution until the next line in the current function is reached or it returns.
# - s(tep): Execute the current line, stop at the first possible occasion (either in a function that is called or in the current function).
# - r(eturn): Continue execution until the current function returns.
# 
# - r(eturn): Continue execution until the current function returns.
# - a(rgs): Print the argument list of the current function.

# In[ ]:


from IPython.core.debugger import set_trace


# In[ ]:


def my_function(x):
    answer = 42
    set_trace()
    answer += x
    return answer

my_function(12)


# ## Example: debuging a NN

# In[ ]:


X = torch.rand((5, 3))
X


# In[ ]:


class MyModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(3, 1)
    
    def forward(self, X):
        # set_trace()
        x = self.lin(X)
        return X

    
model = MyModule()
y_ = model(X)

assert y_.shape == (5, 1), y_.shape


# # Recap - what we learned so far
# - Tensor like numpy
# - No need to calculate derivatives - automatic differentiation!
# - Use `nn.Module` to create your own networks
# - `set_trace` is your friend!

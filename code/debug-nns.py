
# coding: utf-8

# In[1]:

import torch
import torch.nn as nn


# In[2]:


from IPython.core.debugger import set_trace


# In[9]:


class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(3, 1)
        
    def forward(self, X):
        set_trace()
        X = self.lin(X)
        return X


# In[10]:


model = MyModel()


# In[11]:


X = torch.rand((5, 3))
y_pred = model(X)


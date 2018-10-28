import torch

'''
Autograd with torch

'''

# Example 1:
# create a tensor and track all operations on this to get the gradient
# Here we start with an input and define a sequence of operations on this tensor

x = torch.ones(2, 2, requires_grad=True)

y = x + 2

z = y * y * 3

out = z.mean()

#print(z, out)

out.backward()
print(x.grad)


# Example 2:
a = torch.randn(2, 2) # default value of requires_grad is False

a = ((a * 3) / (a - 1)) # element wise multiplication, subtraction and division on tensors is already defined

#print(a.requires_grad)

a.requires_grad_(True) # function that overrides the gradient track requirement

#print(a.requires_grad)

b = (a * a).sum()

#print(b.data)

#print(b.grad_fn)






'''
This is the example from the torch home page

'''
W_h = torch.randn(20, 20, requires_grad=True)
W_x = torch.randn(20, 10, requires_grad=True)

x = torch.randn(1, 10) # stored as a row vector
prev_h = torch.randn(1, 20) # stored as a row vector

h2h = torch.mm(W_h, prev_h.t())
i2h = torch.mm(W_x, x.t())

next_h = h2h + i2h
next_h = next_h.tanh()

loss = next_h.sum()
loss.backward()

#print(W_h.grad)




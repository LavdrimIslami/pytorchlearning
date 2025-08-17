import torch

#NN use back propogation
#torch.autograd computes the gradient descent stuff

x = torch.ones(5) #input tensor
y = torch.ones(3) #expected output
w = torch.randn(5, 3, requires_grad=True)
b = torch.randn(3, requires_grad=True)
z = torch.matmul(x, w) + b
loss = torch.nn.functional.binary_cross_entropy_with_logits(z, y)


#w and b are parameters that need to be optimized

print(f"Gradient function for z = {z.grad_fn}")
print(f"Gradient function for loss = {loss.grad_fn}")


#computing gradients
#compute this with loss.backwards and then retrieve the value

loss.backward()
print(w.grad)
print(b.grad)

#disable gradient tracking

#all tensors with reqgrad true are tracking their history
#when we trained the model and want to apply it to some input we only want to do forward computations

z = torch.matmul(x, w) + b
print(z.requires_grad)

with torch.no_grad():
    z = torch.matmul(x, w) + b
print(z.requires_grad)\

#you can also use detatch

z = torch.matmul(x, w) + b
z_det = z.detach()
print(z_det.requires_grad)



#why would you wanna disable gradient tracking
    #mark parameters in nn as frozen
    #speed up computations when doing forward pass


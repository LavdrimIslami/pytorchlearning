import torch
import numpy as np


#creating a tensor from data directly
data = [[1,2], [3,4]]
x_data = torch.tensor(data)

#creating a tensor from NumPy arrays
np_array = np.array(data)
x_np = torch.from_numpy(np_array)

#creating a tensor from another tensor
x_ones = torch.ones_like(x_data) #retains properties from x_data
print(f"Ones tensor: \n {x_ones} \n")

x_rand = torch.rand_like(x_data, dtype=torch.float)
print(f"Random Tensor: \n {x_rand} \n")


#using random and constant values

#shape is a tuple of tensor dimensions, determines dimensionality of the output tensor

shape = (2,3,)
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)

print(f"Random Tensor: \n {rand_tensor} \n")
print(f"Ones Tensor: \n {ones_tensor} \n")
print(f"Zeros Tensor: \n {zeros_tensor}")

#attributes
tensor = torch.rand(3,4)
print(f"Shape of tensor: {tensor.shape}")
print(f"Datatype of tensor: {tensor.dtype}")
print(f"Device tensor is stored on: {tensor.device}")


#operations on Tensors

#we move our tensor to the current accelorator
if torch.accelerator.is_available():
    tensor = tensor.to(torch.accelerator.current_accelerator())


tensor = torch.ones(4,4)
print(f"First row: {tensor[0]}")
print(f"First column: {tensor[:,0]}")
print(f"Last column: {tensor[..., -1]}")
tensor[:,1] = 0
print(tensor)

#you can join tensors torch.cat to concatenate a sequence of tensors along a given dimensions
#torch.stack also works somewhat

t1 = torch.cat([tensor, tensor, tensor], dim = 1)
print(t1)


#arithmetic ops
#computes matrix multiplacions between 2 tensors

y1 = tensor @ tensor.T
y2 = tensor.matmul(tensor.T)

y3 = torch.rand_like(y1)
torch.matmul(tensor, tensor.T, out=y3)

#computes element wise product, z1 z2 z3 will have the same value
z1 = tensor * tensor
z2 = tensor.mul(tensor)

z3 = torch.rand_like(tensor)
torch.mul(tensor, tensor, out=z3)
print(tensor)


#if u have a single element tensor, you can convert it to a python numerical value with item()

agg = tensor.sum()
agg_item = agg.item()
print(agg_item, type(agg_item))

#operations that store the result into the operand are called in place, denoted by _ 

print(f"{tensor} \n")
tensor.add_(5)
print(tensor)


#tensors on cpu and numpy arrays can share the same memory location
t = torch.ones(5)
print(f"t: {t}")
n = t.numpy()
print(f"n: {n}")

#changing tensor will also change the array
t.add_(1)
print(f"t: {t}")
print(f"n: {n}")


#numpy array to tensor
n = np.ones(5)
t = torch.from_numpy(n)

np.add(n,1,out = n)
print(f"t: {t}")
print(f"n: {n}")
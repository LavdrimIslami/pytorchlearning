import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )
    
    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits
    

#instance of nn, print the structure

model = NeuralNetwork().to(device)
print(model)


#to use the model, pass data in. this will execute forward along with some other stuff
#note: dont call model.forward directly

#returns a 2D tensor with dim=0 corresponding to each output of 10 raw predicted values for each class, dim=1 corresponding to the individual values of each output

#we get prediction probs by passing it through an instance of softmax

X = torch.rand(1,28,28, device=device)
logits = model(X)

pred_probab = nn.Softmax(dim=1)(logits)
y_pred = pred_probab.argmax(1)
print(f"Predicted class: {y_pred}")


input_image = torch.rand(3,28,28)
print(input_image.size())

flatten = nn.Flatten()
flat_image = flatten(input_image)
print(flat_image.size())

#linear layer is a module that applies a linear transformation on the input using its stored weights and biases
layer1 = nn.Linear(in_features=28*28, out_features=20)
hidden1 = layer1(flat_image)
print(hidden1.size())


#non linear activations are what create complex mappings between inputs and outputs
#use relu between lienar layers
print(f"Before relu: {hidden1}\n\n")
hidden1 = nn.ReLU()(hidden1)
print(f"After ReLU: {hidden1}")

#sequential is an ordered container of modules, u can use this to put together a quick network 

seq_modules = nn.Sequential(
    flatten,
    layer1,
    nn.ReLU(),
    nn.Linear(20,10)
)

input_image = torch.rand(3,28,28)
logits = seq_modules(input_image)

#softmax
#last layer of nn returns logits, passed to softwmax
#scaled to values 0 1
#dim param indicates the dimensions along which the values must sum to 1

softmax = nn.Softmax(dim = 1)
pred_probab = softmax(logits)


#model params
#iterate over each parameter, print its size and a preview of its values

print(f"Model structure: {model}\n\n")
for name, param in model.named_parameters():
    print(f"Layer: {name} | Size: {param.size()} | Values : {param[:2]}\n")
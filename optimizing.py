import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor


training_data = datasets.FashionMNIST(
    root = "data",
    train = False,
    download = True,
    transform = ToTensor()
)

test_data = datasets.FashionMNIST(
    root = "data",
    train = False,
    download = True,
    transform = ToTensor()
)

train_dataloader = DataLoader(training_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)

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
    

#instance of nn
model = NeuralNetwork()



#hyperparameters
#adjustable params let you control optimization process
    #number of epochs
    #batch size
    #learning rate

learning_rate = 1e-3
batch_size = 64
epochs = 5

#an epoch consists of the train loop, and validation loop

#loss function measures the degree of dissimilarity of result to target. we wanna minimize that

#loss functions include: MSELoss, NLLLoss, crossentloss

loss_fn = nn.CrossEntropyLoss()


#optimizer: process of adjusting model params to reduce model error

optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate)


#in the training loop, we optimize with zerograd, backprop, and step

def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    #set model to training mode
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        #compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        #backprop
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * batch_size + len(X)
            print(f"Loss: {loss:>7f}   [{current:>5d}/{size:>5d}]")

def test_loop(dataloader, model, loss_fn):
    #evaluate mode
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0 


    #evaluating with torchnograd ensures no gradients are computed during test mode
    #reduces uneccessary computes 
    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}% Avg Loss: {test_loss:>8f} \n")


loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate)

epochs = 10
for t in range (epochs):
    print(f"Epoch {t + 1}\n------------------------")
    train_loop(train_dataloader, model, loss_fn, optimizer)
    test_loop(test_dataloader, model, loss_fn)
print("Done")
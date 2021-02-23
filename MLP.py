import torch
import torchvision.transforms
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST

# fix random seed for reproducibility
seed = 7
torch.manual_seed(seed)
import numpy as np
np.random.seed(seed)

# flatten 28*28 images to a 784 vector for each image
transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),  # convert to tensor
    torchvision.transforms.Lambda(lambda x: x.view(-1))  # flatten into vector
])
trainset = MNIST(".", train=True, download=True, transform=transform)
testset = MNIST(".", train=False, download=True, transform=transform)

# create data loaders
trainloader = DataLoader(trainset, batch_size=128, shuffle=True)
testloader = DataLoader(testset, batch_size=128, shuffle=True)

class_counts = torch.zeros(10, dtype=torch.int32)

for (images, labels) in trainloader:
    for label in labels:
      class_counts[label]+=1

print(class_counts)

from MLPmodel import Perceptron
from torch import nn

device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(device)
model = Perceptron(784, 784, 10).to(device) # build the model

from torch import optim
# define the loss function and the optimiser
loss_function = nn.CrossEntropyLoss()
optimiser = optim.Adam(model.parameters())

# the epoch loop
for epoch in range(10):
    running_loss = 0.0
    for data in trainloader:
        # get the inputs
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        # zero the parameter gradients
        optimiser.zero_grad()

        # forward + loss + backward + optimise (update weights)
        outputs = model(inputs)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimiser.step()

        # keep track of the loss this epoch
        running_loss += loss.item()
    print("Epoch %d, loss %4.2f" % (epoch, running_loss))
print('**** Finished Training ****')

model.eval()

# Compute the model accuracy on the test set
correct = 0
total = 0
class_correct = torch.zeros(10)
class_total = torch.zeros(10)
for data in testloader:
        # get the inputs
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        outputs = model(inputs)
        for i in range(len(labels)):
          #print(i)
          index =((outputs[i] == torch.max(outputs[i])).nonzero())
          if(labels[i]==index):
            correct +=1
            class_correct[index] += 1
          total += 1
          class_total[index] += 1
print(correct)
print('Test Accuracy: %2.2f %%' % ((100.0 * correct) / total))

for i in range(10):
    print('Class %d accuracy: %2.2f %%' % (i, 100.0*class_correct[i] / class_total[i]))
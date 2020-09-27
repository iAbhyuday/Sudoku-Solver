import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import numpy as np
from torch.autograd import Variable
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision import transforms
from LeNet import LeNet
from torchvision.datasets import MNIST
if torch.cuda.is_available():
    dev = "cuda"
else:
    dev = "cpu"
data_transform = transforms.ToTensor()
resize_transform = transforms.Resize((28,28))
grayscale_transform = transforms.Grayscale()
norm_transform = transforms.Normalize(0,255)

device = torch.device(dev)

transform = transforms.Compose([grayscale_transform,resize_transform,data_transform,norm_transform])

dataset = datasets.ImageFolder(root="datasets",transform=transform)


# train_data = MNIST(root="./data",train=True,download=False,transform=data_transform)

# test_data = MNIST(root="./data",train=False,download=False,transform=data_transform)


batch_size = 16

train_loader = DataLoader(dataset,batch_size=8,shuffle=True)
test_loader = DataLoader(dataset,batch_size=batch_size,shuffle=True)

model = LeNet().cuda()
print("Model initialized !")
optimizer = optim.Adam(params=model.parameters(),lr=0.001)
criterion = nn.CrossEntropyLoss()

def train(n_epochs):
    print("Training Started !")
    for epoch in range(1,n_epochs+1):
        
        running_loss=0
        for batch_i,data in enumerate(train_loader):

            inputs,labels = data[0].cuda(),data[1].cuda()
            

            inputs,labels = Variable(1-inputs),Variable(labels)

            optimizer.zero_grad()

            logits = model(inputs)
          
            loss = criterion(logits,labels)

            loss.backward()

            optimizer.step()


            running_loss+=loss.item()

            if batch_i%100==99:
                print('Epoch: {}, Batch: {}, Avg. Loss: {}'.format(epoch, batch_i+1, running_loss/100),end=" ")
                model.eval()
                total=0
                correct=0
                val_rl=0
                for data in test_loader:
                    image,label = data[0].cuda(),data[1].cuda()
                    out = model(1-image)
                    val_loss = criterion(out,label)
                   
                    val_rl+=val_loss.item()
                    _,predicted = torch.max(out.data,1)
                    total+=label.size(0)
                    correct+= (predicted==label).sum().item()
                
                print("Val_acc : ",100*correct/total,end="")
                print(" Val_loss : ",val_rl/100)
                val_rl=0
                model.train()
              
                running_loss = 0.0
    
    print("Finished Training !")

# Train upto 10 epochs
train(10)

model.to("cpu")
torch.save(model,"sudoku.pth")
print("Model saved !")


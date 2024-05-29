from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader            #data loader
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")      #check if u have a gpu that you can use
#print(DEVICE)

training_data = datasets.FashionMNIST(
    root="data",
    train=True,                                      #all data
    download=True,
    transform=ToTensor()
)

testing_data = datasets.FashionMNIST(
    root="data",     
    train=False,
    download=True,
    transform=ToTensor()
)

loaders = {                                #loading data into the model
    "train": DataLoader(
        training_data,
        batch_size=50,
        shuffle=True,
        num_workers=1 ),
    "test": DataLoader(
    testing_data,
    batch_size=50,
    shuffle=True,
    num_workers=1 ),
}

class CNN(nn.Module):
    def __init__(self):
        super(CNN,self).__init__()

        self.conv1 = nn.Conv2d(1,10,kernel_size=5)           #Applies a 2D convolution over an input signal composed of several input planes.
        self.conv2 = nn.Conv2d(1,10,kernel_size=5)           #complex maths i do not know
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320,50)                         #ai architecture multiplies to 320 neurons
        self.fc2 = nn.Linear(50,10)

    def forward(self,x):
        x = F.relu(F.max_pool2d(self.conv1(x),2))                  # applies rectified linear unit to input data
        x = F.relu(F.max_pool2d(self.conv2_drop((x),2)))          #ai conv layer
        x = x.view(-1,320)                                       #returns the tensor data feeds data to linear layer
        x = x.relu(self.fc1(x))                                 #reapplies rectified linear unit
        x = F.dropout(x,training=self.training)             # done to prevent overfitting
        x = self.fc2(x)
        return F.softmax(x)
    

model = CNN().to(DEVICE)
optimizer = optim.Adam(model.parameters(),lr=0.001)

loss_fn = nn.CrossEntropyLoss()

def train(EPOCH):
    model.train()                           #puts model into training mode
    for batch_idx, (data,target) in enumerate(loaders["train"]):
        data, target = data.to(DEVICE), data.to(DEVICE)                     #keeps data on cpu or gpu
        optimizer.zero_grad()
        output = model(data)                          #make a prediction with data 
        loss= loss_fn(output, target)                  #calculate loss
        loss.backward()                               #back propagration 
        optimizer.step()       #upate based on current gradient
        if batch_idx % 20 == 0:
            print(f"Train epoch {EPOCH} [{batch_idx *len(data)}/{len(loaders['train'].dataset)} ({100. * batch_idx /len(loaders['train']):.0f}%)]\t{loss.item():.6f}")   #didnt understand any of this mainly formatting
    
def test():
    model.eval()
    test_loss=0
    correct=0

    with torch.no_grad():
        for data, target in loaders["test"]:
            data, target = data.to(DEVICE), data.to(DEVICE)                           
            output = model(data)                             
            test_loss += loss_fn(output,target).item()                              
            prediction= output.argmax(dim=1,keepdim= True)                                 #checks predictions
            correct += prediction.eq(target.view_as(prediction)).sum().item()                #adds to correct when a prediciton is right

            test_loss/= len(loaders["test"].dataset)
            print(f"\n Test set : Average Loss: {test_loss: .4f}, accuracy: {correct}/{len(loaders['test'].dataset)} ({100. * correct / len(loaders['test'].dataset):0.f}%\n)")

for EPOCH in range(1,11):
    train(EPOCH)
    test()
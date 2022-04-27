#import

import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'


#create Network
class NN(nn.Module):
    def __init__(self,input_size,num_classes):
        super(NN,self).__init__()
        self.fc1 = nn.Linear(input_size, 50)
        self.fc2 = nn.Linear(50,num_classes)

    def forward(self,x):
        x= F.relu(self.fc1(x))
        x= self.fc2(x)

        return x

class CNN(nn.Module):
    def __init__(self,in_channel=1,num_classes=10):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1,out_channels=8,kernel_size=(3,3),stride=(1,1),padding=(1,1)) #same convolution
        self.pool = nn.MaxPool2d(kernel_size=(2,2),stride=(2,2))
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.fc1 = nn.Linear(16*7*7, num_classes)

    def forward(self,x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x= x.reshape(x.shape[0], -1)
        x= self.fc1(x)

        return x

model = CNN()
x = torch.randn(64,1,28,28)
print(model(x).shape)

# model = NN(784,10)
# x = torch.randn(64,784)
# print(model(x).shape)

#Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#hyperparmeters
in_channel = 1
num_classes = 10
#learning_rate =0.001
#batch_size = 64
num_epochs = 1

#Load Data
train_dataset=datasets.MNIST(root='dataset/',train=True,transform=transforms.ToTensor(),download=True)

test_dataset=datasets.MNIST(root='dataset/',train=False,transform=transforms.ToTensor(),download=True)
# test_loader = DataLoader(dataset=test_dataset,batch_size=batch_size,shuffle=True)





batch_sizes =[256]
learning_rates = [0.001]
classes = ['0','1','2','3','4','5','6','7','8','9']

for batch_size in batch_sizes:
    for learning_rate in learning_rates:
        step = 0

        # Initialize network
        model = CNN(in_channel=in_channel, num_classes=num_classes)
        model.to(device)
        model.train()
        train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
        # loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        writer = SummaryWriter(f'runs/MNIST/MiniBatchSize {batch_size} LR {learning_rate}')

        #train
        for epoch in range(num_epochs):
            losses = []
            accuracies =[]

            for batch_idx,(data,targets) in enumerate(train_loader):
                #get data to cuda if possible
                data = data.to(device=device)
                targets = targets.to(device=device)

                #forward
                scores = model(data)
                loss = criterion(scores,targets)
                losses.append(loss.item())

                #backward
                optimizer.zero_grad()
                loss.backward()
                #gradinet descent or adam step
                optimizer.step()

                #calculate 'running' training accuracy
                features = data.reshape(data.shape[0],-1)
                img_grid = torchvision.utils.make_grid(data)
                _,predictions = scores.max(1)
                num_correct = (predictions==targets).sum()
                running_train_acc = float(num_correct)/float(data.shape[0])
                accuracies.append(running_train_acc)

                #plot things to tensorboard
                class_labels = [classes[label] for label in predictions]
                writer.add_image('mnist_images', img_grid)
                writer.add_histogram('fc1', model.fc1.weight)
                writer.add_scalar('Training Loss',loss,global_step=step)
                writer.add_scalar('Training accuracy',running_train_acc,global_step=step)
                if batch_idx ==230:
                    writer.add_embedding(features,metadata=class_labels,label_img=data,global_step=batch_idx)

                step += 1

            writer.add_hparams({'lr':learning_rate, 'bsize': batch_size},
                               {'accuracy': sum(accuracies) / len(accuracies),
                                                 'loss': sum(losses) / len(losses)})

            print(f'Mean Loss this epoch was {sum(losses)/len(losses)}')

#check accuarcy on training , test
def check_accuarcy(loader,model):
    if loader.dataset.train:
        print("checking on accuracy on training data")
    else:
        print("checking on accuracy on test data")
    num_correct = 0
    num_sample = 0
    model.eval()

    with torch.no_grad():
        for x,y in loader:
            x = x.to(device = device )
            y = y.to(device=device)

            scores = model(x)
            _,predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_sample += predictions.size(0)

        print(f"got{num_correct}/{num_sample} with accuarcy {float(num_correct)/float(num_sample)*100:.2f}")

    model.train()


# check_accuarcy(train_loader,model)
# check_accuarcy(test_loader,model)

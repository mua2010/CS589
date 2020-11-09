from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

import numpy as np
import torch.utils.data as utils
import matplotlib.pyplot as plt

# The parts that you should complete are designated as TODO
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        # TODO: define the layers of the network
        _dict = {
            "in_channels": 1,
            "out_channels": 32, 
            "kernel_size": 3,
            "stride": 1,
            "padding": 0
        }
        self.conv_one = nn.Conv2d(**_dict)
        _dict = {
            "in_channels": 32,
            "out_channels": 64, 
            "kernel_size": 3,
            "stride": 1,
            "padding": 0
        }
        self.conv_two = nn.Conv2d(**_dict)
        self.mp_drop = nn.Dropout2d(p=0.25)
        self.fc1 = nn.Linear(64*12*12, 128)
        self.fc1_drop = nn.Dropout2d(p=0.5)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        # TODO: define the forward pass of the network using the layers you defined in constructor
        x = F.relu(self.conv_one(x))
        x = F.relu(self.conv_two(x))
        x = F.max_pool2d(x, 2)
        x = self.mp_drop(x)
        # breakpoint()
        # x = x.flatten()
        x = x.reshape(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc1_drop(x)
        return self.fc2(x)


def train(model, device, train_loader, optimizer, epoch):
    model.train()
    correct = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0: #Print loss every 100 batch
            print('Train Epoch: {}\tLoss: {:.6f}'.format(
                epoch, loss.item()))
    accuracy = test(model, device, train_loader)
    return accuracy

def test(model, device, test_loader):
    model.eval()
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    accuracy = 100. * correct / len(test_loader.dataset)

    return accuracy


def main():
    torch.manual_seed(1)
    np.random.seed(1)
    # Training settings
    use_cuda = False # Switch to False if you only want to use your CPU
    learning_rate = 0.01
    NumEpochs = 10
    batch_size = 32

    device = torch.device("cuda" if use_cuda else "cpu")

    train_X = np.load('../../Data/X_train.npy')
    train_Y = np.load('../../Data/y_train.npy')

    test_X = np.load('../../Data/X_test.npy')
    test_Y = np.load('../../Data/y_test.npy')

    train_X = train_X.reshape([-1,1,28,28]) # the data is flatten so we reshape it here to get to the original dimensions of images
    test_X = test_X.reshape([-1,1,28,28])

    # transform to torch tensors
    tensor_x = torch.tensor(train_X, device=device)
    tensor_y = torch.tensor(train_Y, dtype=torch.long, device=device)

    test_tensor_x = torch.tensor(test_X, device=device)
    test_tensor_y = torch.tensor(test_Y, dtype=torch.long)

    train_dataset = utils.TensorDataset(tensor_x,tensor_y) # create your datset
    train_loader = utils.DataLoader(train_dataset, batch_size=batch_size) # create your dataloader

    test_dataset = utils.TensorDataset(test_tensor_x,test_tensor_y) # create your datset
    test_loader = utils.DataLoader(test_dataset) # create your dataloader if you get a error when loading test data you can set a batch_size here as well like train_dataloader

    model = ConvNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train_acc_list = list()
    test_acc_list = list()

    for epoch in range(NumEpochs):
        train_acc = train(model, device, train_loader, optimizer, epoch)
        train_acc_list.append(train_acc)
        print('\nTrain set Accuracy: {:.1f}%\n'.format(train_acc))
        test_acc = test(model, device, test_loader)
        test_acc_list.append(test_acc)
        print('\nTest set Accuracy: {:.1f}%\n'.format(test_acc))
        #TODO: Store train_acc and test_acc in an array to plot later.
        
    torch.save(model.state_dict(), "mnist_cnn.pt")

    #TODO: Plot train and test accuracy vs epoch
    plt.figure("CNN train and test accuracy vs epoch")
    plt.plot(list(range(NumEpochs)), train_acc_list, c='b', label="train accuracy")
    plt.plot(list(range(NumEpochs)), test_acc_list, c='r', label="test accuracy")
    plt.ylabel("accuracy")
    plt.xlabel("# epoch")
    # plt.xlim(0,15)
    plt.legend(loc=0)
    plt.savefig('../Figures/q1.2b_CNN_train_test_accuracy_vs_epoch.png')

if __name__ == '__main__':
    main()

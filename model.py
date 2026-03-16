import ray
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import List
from collections import OrderedDict



class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size= 5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320,50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.max_pool2d(x, 2)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.conv2_drop(x)
        x = F.max_pool2d(x,2)
        x = F.relu(x)
        x = x.view(-1, x.shape[1]*x.shape[2]*x.shape[3])
        x = self.fc1(x)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return  x

# class Net(nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
#         self.fc1 = nn.Linear(784, 256)
#         self.fc2 = nn.Linear(256, 128)
#         self.fc3 = nn.Linear(128, 10)
    
#     def forward(self, x):
#         x = x.view(-1, 28*28)  # Flatten the input
#         x = torch.relu(self.fc1(x))
#         x = torch.relu(self.fc2(x))
#         x = self.fc3(x)
#         return x
    





def train(net, trainloader, clientManager, client_updates: int, device, verbose=False):
    """Train the network on the training set."""

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        params = net.parameters(), 
        lr = 0.01, 
        momentum = 0.9
        )
    
    net.train()

    local_iter = 0
    end = False
    loss = 0.0
    correct = 0
    total = 0
    actual_epoch = 0

    local_comp_energy_consumed = 0.0
    local_train_time = 0.0

    for epoch in range(1000):
        for i, batch in enumerate(trainloader):
            images, labels = batch["image"].to(device), batch["label"].to(device)

            optimizer.zero_grad()
            outputs = net(images)
            batch_loss = criterion(outputs, labels)
            batch_loss.backward()
            optimizer.step()

            # Metrics
            loss += batch_loss * labels.size(0)
            num_samples_batch = labels.size(0)
            total += num_samples_batch # the number of samples used
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
            # FINE ESECUZIONE BATCH
            
            # computation of energy consumption and training time for batch
            batchEnergyConsumed = clientManager.computeEnergyComputation(num_samples_batch) # samples batch: 20, energy computation for batch: 0.00025088 J
            batchTrainTime = clientManager.computeTrainTimeComputation(num_samples_batch)   # Batch samples: 20 - Training time = 0.0025088 s

            print(f"Batch samples: {num_samples_batch}, Energy computation for batch: {batchEnergyConsumed} J, Training time = {batchTrainTime} s")
            local_comp_energy_consumed += batchEnergyConsumed
            local_train_time += batchTrainTime

            local_iter += 1
            if local_iter >= client_updates:
                end = True
                print(f"Local iteration: {local_iter}")
                actual_epoch = epoch + 1
                #exp_lr_scheduler(actual_epoch, scheduler, 1)
                break
        if end: break
        actual_epoch = epoch + 1
        # exp_lr_scheduler(actual_epoch, scheduler, 1)
    
    # print(f"local update energy consumed: {local_update_energy_consumed}")
    # print(f"train time: {local_update_train_time}")

    return local_comp_energy_consumed, local_train_time





    # for epoch in range(epochs):
    #     correct, total, epoch_loss = 0, 0, 0.0
    #     for batch in trainloader:
    #         images, labels = batch["image"].to(device), batch["label"].to(device)
    #         optimizer.zero_grad()
    #         outputs = net(images)
    #         loss = criterion(outputs, labels)
    #         loss.backward()
    #         optimizer.step()
    #         # Metrics
    #         epoch_loss += loss
    #         total += labels.size(0)
    #         correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
    #     epoch_loss /= len(trainloader.dataset)
    #     epoch_acc = correct / total
    #     if verbose:
    #         print(f"Epoch {epoch+1}: train loss {epoch_loss}, accuracy {epoch_acc}")


def test(net, testloader, device):
    """Evaluate the network on the entire test set."""
    criterion = torch.nn.CrossEntropyLoss()
    correct, total, loss = 0, 0, 0.0
    net.eval()
    with torch.no_grad():
        for batch in testloader:
            images, labels = batch["image"].to(device), batch["label"].to(device)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    loss /= len(testloader.dataset)
    accuracy = correct / total
    return loss, accuracy


def set_parameters(net, parameters: List[np.ndarray]):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)


def get_parameters(net) -> List[np.ndarray]:
    return [val.cpu().numpy() for _, val in net.state_dict().items()]


def get_state_model(model):
    return {key: value.clone() for key, value in model.state_dict().items()}


# Function to compute the size of the updated model parameters
def get_updated_size(original_state, updated_state, precision_bits=32):
    total_bits = 0

    # Calculate the number of bits for the parameters that have changed
    for orig_param, updated_param in zip(original_state.values(), updated_state.values()):
        if not torch.equal(orig_param, updated_param):
            # Compute the number of elements (parameters)
            num_elements = orig_param.numel()
            # Calculate total bits
            total_bits += num_elements * precision_bits

    return total_bits
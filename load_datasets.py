import os

import torch
import flwr_datasets
from flwr_datasets.partitioner import PathologicalPartitioner

from typing import Tuple

from torchvision import datasets, transforms
from torch.utils.data import DataLoader




def load_datasets(config, partition_id: int) -> Tuple[DataLoader, DataLoader]:

    print(f"LOADING PARTITION ID: {partition_id}")

    partitioner = PathologicalPartitioner(
            num_partitions= config['NUM_CLIENTS'],
            partition_by="label",
            num_classes_per_partition=1,
            class_assignment_mode="first-deterministic"
            )
    
    fds = flwr_datasets.FederatedDataset(dataset="mnist", partitioners={"train": partitioner}, seed=config['SEED'])
    
    partition = fds.load_partition(partition_id, "train")

    pytorch_transforms = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])
    
    def apply_transforms(batch):
        # Instead of passing transforms to CIFAR10(..., transform=transform)
        # we will use this function to dataset.with_transform(apply_transforms)
        # The transforms object is exactly the same
        batch["image"] = [pytorch_transforms(image) for image in batch["image"]]
        return batch

    partition = partition.with_transform(apply_transforms)


    trainloader = DataLoader(
        partition, batch_size=config['BATCH_SIZE'], shuffle=False
    )

    testset = fds.load_split("test").with_transform(apply_transforms)
    testloader = DataLoader(testset, batch_size=config['BATCH_SIZE'])

    return trainloader, testloader



def load_dataloaders(config):
    """Load dataloaders for training and testing."""

    trainloaders = []
    testloader = None

    for partition_id in range(config['NUM_CLIENTS']):
        trainloader, _ = load_datasets(config, partition_id)
        trainloaders.append(trainloader)
        print(f"len trainloader for client {partition_id}: {len(trainloader.dataset)} samples")
    
    testloader = load_datasets(config, partition_id=0)[1]

    return trainloaders, testloader
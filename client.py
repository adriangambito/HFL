import torch

from flwr.client import Client, NumPyClient
from flwr.common import Metrics, Context

from model import Net, train, get_parameters, set_parameters, get_state_model, get_updated_size
from load_datasets import load_datasets

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class FlowerClient(NumPyClient):
    def __init__(self, id, net, trainloader, clientManager, device, context: Context):
        self.client_state = (
            context.state
        )

        self.id = id
        self.net = net  # already set to device
        self.trainloader = trainloader
        self.clientManager = clientManager
        self.device = device

    def get_parameters(self, config):
        return get_parameters(self.net)


    def fit(self, parameters, config):
        set_parameters(self.net, parameters)
        print(f"Flower client {self.id} with ClientManager {self.clientManager.client_id} starting training...")
        initial_state = get_state_model(self.net)
        localUpdateEnergyComputation, localUpdateTrainTime = train(self.net, self.trainloader, self.clientManager, client_updates=config['client_updates'], device=self.device)
        # print(f"localUpdateEnergyComputation: {localUpdateEnergyComputation} J")
        # print(f"localUpdateTrainTime: {localUpdateTrainTime} s")
        update_state = get_state_model(self.net)

        # compute the difference between initial_state and update_state
        update_size_bits = get_updated_size(initial_state, update_state, precision_bits=32)
        print(f"Client {self.id} update size: {update_size_bits} bits")

        localUpdateEnergyCommunication = self.clientManager.computeEnergyCommunication(update_size_bits)
        print(f"localUpdateEnergyCommunication: {localUpdateEnergyCommunication} J")

        # Compute number of communications
        self.clientManager.setNumCommunications(1)
        clientCommunications = self.clientManager.getNumCommunications()
        print(f"Client {self.id} number of communications: {clientCommunications}")

        return get_parameters(self.net), len(self.trainloader), {"client_id": self.id,
                                                                 "consumedEnergyComputation": localUpdateEnergyComputation,
                                                                 "consumedEnergyCommunication": localUpdateEnergyCommunication,
                                                                 "trainTimeComputation": localUpdateTrainTime,
                                                                 "numCommunications": clientCommunications}


    # def evaluate(self, parameters, config):
    #     set_parameters(self.net, parameters)
    #     loss, accuracy = test(self.net, self.valloader, device=self.device)
    #     return float(loss), len(self.valloader), {"accuracy": float(accuracy)}
    

def get_client_fn(config, trainloaders, clientManagers):
    def client_fn(context: Context) -> Client:
        """Create a Flower client representing a single organization."""
        device = config.get("DEVICE", "cpu")
        # Load model
        net = Net().to(device)

        # Load data (CIFAR-10)
        # Note: each client gets a different trainloader/valloader, so each client
        # will train and evaluate on their own unique data partition
        # Read the node_config to fetch data partition associated to this node
        partition_id = context.node_config["partition-id"]
        print(f"Creating client for partition id: {partition_id}")
        trainloader = trainloaders[partition_id]
        clientManager = clientManagers[partition_id]

        # Create a single Flower client representing a single organization
        # FlowerClient is a subclass of NumPyClient, so we need to call .to_client()
        # to convert it to a subclass of `flwr.client.Client`
        return FlowerClient(partition_id, net, trainloader, clientManager, device, context).to_client()
    
    return client_fn
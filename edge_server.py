import random

from flwr.common import Context
from flwr.common import (
    ndarrays_to_parameters,
)
from flwr.server import ServerConfig, ServerAppComponents
from flwr.common import Metrics

from model import Net, get_parameters, set_parameters, test
from load_datasets import load_datasets
from strategy import CustomFedAvg

class EdgeServer:
    def __init__(self, server_id):
        self.server_id = server_id
        self.num_clients = None  # Number of clients assigned to this edge server
        self.clients = []  # List to hold the devices (clients) assigned to this edge server
        
        # Add CPU power (in GHz), memory (in GB), and other configurations
        self.cpu_power = random.uniform(1.5, 3.5)  # Random CPU power between 1.5GHz and 3.5GHz
        self.memory = random.randint(4, 32)  # Random memory between 4GB and 32GB
        self.storage = random.randint(50, 500)  # Random storage between 50GB and 500GB

        # Transmitter power for edge-to-cloud communication in watts
        min_power, max_power = 0.8, 1.10  # in Watt
        self.transmitter_power = random.uniform(min_power, max_power)  # Example: Higher power compared to device-to-edge

        # Channel capacity)
        self.channel_capacity = 1e7  # Fixed 10 Mbps for edge-to-cloud communication

    def __repr__(self):
        return (f"EdgeServer(server_id={self.server_id}, "
                f"cpu_power={self.cpu_power:.6f}, "
                f"memory={self.memory:.6f}, "
                f"storage={self.storage:.6f} GB, "
                f"num_clients={self.num_clients}, "
                f"clients={self.clients} "
                )

    def set_transmitter_power(self, power):
        """
        Update the transmitter power for edge-to-cloud communication.
        """
        self.transmitter_power = power

    def get_transmitter_power(self):
        """
        Get the transmitter power for edge-to-cloud communication.
        """
        return self.transmitter_power
    

    def assign_clients(self, clients):
        """
        Assign clients (clients) to the edge server.
        """
        self.clients = clients

    def add_client(self, client):
        """
        Add a single client to the edge server.
        """
        self.clients.append(client)
    

    def compute_cloud_communication_energy(self, size_model_bits):
        """
        Compute energy for transmitting data to the cloud server.
        """
        communicationLatency = size_model_bits / self.channel_capacity
        energyCommConsumed = self.transmitter_power * communicationLatency
        return energyCommConsumed

    def get_clients(self):
        """
        Return the list of clients assigned to this edge server.
        """
        return self.clients

    def update_configuration(self, cpu_power=None, memory=None, storage=None):
        """
        Update the edge server's configuration.
        """
        if cpu_power is not None:
            self.cpu_power = cpu_power
        if memory is not None:
            self.memory = memory
        if storage is not None:
            self.storage = storage

    def get_configuration(self):
        """
        Return the current configuration of the edge server.
        """
        return {
            "CPU Power (GHz)": self.cpu_power,
            "Memory (GB)": self.memory,
            "Storage (GB)": self.storage
        }

    def get_server_id(self):
        return self.server_id
    
    def set_num_clients(self, num_clients):
        """
        Set the number of clients assigned to this edge server.
        """
        self.num_clients = num_clients

    def get_num_clients(self):
        return len(self.clients)




def get_on_fit_config(config):
    def on_fit_config(server_round: int) -> Metrics:
        """Adjusts hyperparameters based on current round."""

        print("\033[34mON_FIT_CONFIG\033[0m")

        client_updates = config.get("K1", 1)
        return {"client_updates": client_updates}
    
    return on_fit_config


def get_evaluate_fn(testloader, device):
    """Return a callback that evaluates the global model"""
    def evaluate(server_round, parameters_ndarrays, config):
        """Evaluate global model using provided centralised testset."""

        print("\033[34mCENTRALIZED EVALUATE\033[0m")
        net = Net()
        set_parameters(net, parameters_ndarrays)
        net.to(device)
        loss, accuracy = test(net, testloader, device)

        return loss, {"cen_accuracy": accuracy}
    
    return evaluate


def get_server_fn(config, global_params):
    def server_fn(context: Context):
   
        num_rounds = config.get("ROUNDS", 5)
        fraction_fit = 1

        # Initialize model parameters
        # model = Net()
        # ndarrays = get_parameters(model)
        # parameters = ndarrays_to_parameters(ndarrays)

        # Load global test set
        _, testloader = load_datasets(config, partition_id=0)

        
        # Create FedAvg strategy
        strategy = CustomFedAvg(
            config = config,
            fraction_fit=1.0,  # Sample 100% of available clients for training
            fraction_evaluate=0.0,  # Sample 50% of available clients for evaluation
            # min_fit_clients=10,  # Never sample less than 10 clients for training
            # min_evaluate_clients=5,  # Never sample less than 5 clients for evaluation
            # min_available_clients=10,  # Wait until all 10 clients are available
            initial_parameters=global_params,
            on_fit_config_fn=get_on_fit_config(config),
            evaluate_fn=get_evaluate_fn(testloader, device="cpu"),
        )


        serverconfig = ServerConfig(num_rounds=config['K2'])

        return ServerAppComponents(strategy=strategy, config=serverconfig)
    
    return server_fn
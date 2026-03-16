import os
os.environ["RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO"] = "0"


import yaml

import torch
import flwr
from flwr.client import ClientApp
from flwr.server import ServerApp

# from client import get_client_fn
# from server import get_server_fn

from flwr.simulation import run_simulation

from utils import set_seed



from model import Net, set_parameters, test
from flwr.common import FitRes, FitIns, EvaluateIns, Parameters, parameters_to_ndarrays
# from load_datasets import load_datasets
from hfl import hfl_simulation


def main():

    # *CONFIGURATION SETTINGS*
    ################################################################################ 
    # Path Config file yaml
    conffile_path = os.path.join(os.path.dirname(__file__), 'Config', 'config.yaml')

    # Carica il file YAML
    with open(conffile_path, 'r') as f:
        config = yaml.safe_load(f)
    
    config['DEVICE'] = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_CLIENTS = config['NUM_CLIENTS']
    BATCH_SIZE = config['BATCH_SIZE']
    SEED = config['SEED']

    print(f"Training on {config['DEVICE']}")
    print(f"Flower {flwr.__version__} / PyTorch {torch.__version__}")
    ################################################################################ 


    # *MAKE EXPERIMENTS REPRODUCIBLE
    ################################################################################ 
    set_seed(SEED)
    ################################################################################ 
    


    # *EXPERIMENTS LOGS SETUP*
    ################################################################################

    ################################################################################


    # *HIERARCHICAL FEDERATED LEARNING SIMULATION*
    ################################################################################
    hfl_simulation(config)

    ################################################################################ 


    # Create the ClientApp
    # client = ClientApp(client_fn=get_client_fn(config))


    # Create the ServerApp
    # server = ServerApp(server_fn=get_server_fn(config))



    # Specify the resources each of your clients need
    # By default, each client will be allocated 1x CPU and 0x GPUs
    backend_config = {"client_resources": {"num_cpus": 1, "num_gpus": 0.0}}

    # When running on GPU, assign an entire GPU for each client
    if config['DEVICE'] == "cuda":
        backend_config = {"client_resources": {"num_cpus": 1, "num_gpus": 1.0}}
        # Refer to our Flower framework documentation for more details about Flower simulations
        # and how to set up the `backend_config`

    # Run simulation
    # run_simulation(
    #     server_app=server,
    #     client_app=client,
    #     num_supernodes=NUM_CLIENTS,
    #     backend_config=backend_config,
    # )


    # Load global test set
    # _, _, testloader = load_datasets(config, partition_id=0)
    # global_model_parameters= parameters_to_ndarrays(server._strategy.final_parameters)
    # model = Net()
    # set_parameters(model, global_model_parameters)
    # loss, accuracy = test(model, testloader, device="cpu")
    # print(f"Final centralized test set results - Loss: {loss:.4f}, Accuracy: {accuracy:.2f}%")


if __name__ == "__main__":
    main()
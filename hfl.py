from flwr.client import ClientApp
from flwr.server import ServerApp
from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays
from flwr.simulation import run_simulation
from flwr.server.strategy.aggregate import aggregate

from client import get_client_fn
from client_manager import ClientManager
from edge_server import get_server_fn
from model import Net, set_parameters, test
from utils import initialization_model
from hfl_clustering import hfl_clustering
from load_datasets import load_dataloaders


def hfl_simulation(config):



    # * INITIALIZATION MODEL AND DATASETS*
    ################################################################################
    model, init_params = initialization_model(config)
    trainloaders, testloader = load_dataloaders(config)
    ################################################################################

    # * SETUP EDGE SERVERS AND CLIENTS *
    ################################################################################
    clusters = hfl_clustering(config)
    ################################################################################

    # TODO: creare un cloud.py per gestire il global server
    global_results = []
    global_params = ndarrays_to_parameters(init_params)

    accuracies = []
    losses = []

    totalCompEnergy = 0.0
    totalCommEnergy = 0.0
    totalTrainTime = 0.0
    totalNumCommunications = 0


    for global_round in range(config['GLOBAL_ROUNDS']):
        print(f"\n--- GLOBAL ROUND {global_round+1} / {config['GLOBAL_ROUNDS']} ---\n")

        # Process each edge server
        for cluster_id, cluster in clusters.items():
            # print(f"cluster_id: {cluster_id} ")
            # print(f"cluster: {cluster}")
            # print(f"Number of clients in cluster: {len(cluster)}")

            print(f"Processing Edge Server {cluster_id+1} / {len(clusters)}")

            ClusterCompEnergy = 0.0
            ClusterCommEnergy = 0.0
            ClusterTrainTime = 0.0
            ClusterNumCommunications = 0


            clusterTrainloaders = []
            clientsManagers = []

            for client in cluster:
                trainloader = trainloaders[client.client_id]
                clusterTrainloaders.append(trainloader)
                clientsManagers.append(client)
                print(f"len trainloader for client {client.client_id}: {len(trainloader.dataset)} samples")
                print(f"Num batches trainloader: {len(trainloader)}")
            print(f"Number of clients in cluster: {len(clusterTrainloaders)}")
            
            # Create the ClientApp
            client = ClientApp(client_fn=get_client_fn(config, clusterTrainloaders, clientsManagers))


            # Create the ServerApp
            server = ServerApp(server_fn=get_server_fn(config, global_params))



            # Specify the resources each of your clients need
            # By default, each client will be allocated 1x CPU and 0x GPUs
            backend_config = {"client_resources": {"num_cpus": 1, "num_gpus": 0.0}}

            # When running on GPU, assign an entire GPU for each client
            if config['DEVICE'] == "cuda":
                backend_config = {"client_resources": {"num_cpus": 1, "num_gpus": 1.0}}
                # Refer to our Flower framework documentation for more details about Flower simulations
                # and how to set up the `backend_config`

            # Run simulation
            run_simulation(
                server_app=server,
                client_app=client,
                num_supernodes=len(cluster),
                backend_config=backend_config,
            )

            edge_model_parameters = parameters_to_ndarrays(server._strategy.final_parameters)
            edge_total_samples = server._strategy.total_samples
            clientsMetricsTracker = server._strategy.clientsMetricsTracker
            global_results.append((edge_model_parameters, edge_total_samples))

            for client_id in clientsMetricsTracker:
                ClusterCompEnergy += clientsMetricsTracker[client_id]['consumedEnergyComputation']
                ClusterCommEnergy += clientsMetricsTracker[client_id]['consumedEnergyCommunication']
                ClusterTrainTime += clientsMetricsTracker[client_id]['trainTimeComputation']
                ClusterNumCommunications += clientsMetricsTracker[client_id]['numCommunications']

            print(f"Edge Server {cluster_id+1}:\n \
                  - Total Computation Energy Consumed: {ClusterCompEnergy} J \n \
                  - Total Communication Energy Consumed: {ClusterCommEnergy} J \n \
                  - Total Training Time: {ClusterTrainTime} s \n \
                  - Total Number of Communications: {ClusterNumCommunications} \n")

            totalCompEnergy += ClusterCompEnergy
            totalCommEnergy += ClusterCommEnergy
            totalTrainTime += ClusterTrainTime
            totalNumCommunications += ClusterNumCommunications


        print("RESULTS")
        global_parameters_aggregated = aggregate(global_results)    # ndarray format
        global_model = Net().to(config['DEVICE'])
        set_parameters(global_model, global_parameters_aggregated)  # Update model with the latest parameters ndarrays
        global_loss, global_accuracy = test(global_model, testloader, device="cpu")
        accuracies.append(global_accuracy)
        losses.append(global_loss)
        print(f"Global model test set results - Loss: {global_loss:.4f}, Accuracy: {global_accuracy:.2f}%")
        
        global_params = ndarrays_to_parameters(global_parameters_aggregated)
        global_results.clear()



        if global_accuracy >= config['TARGET_ACCURACY']:
            print(f"Target accuracy {config['TARGET_ACCURACY']}% reached. Stopping training.")

            print(f"\n=== FINAL RESULTS ===\n \
                  - Total Computation Energy Consumed: {totalCompEnergy} J \n \
                  - Total Communication Energy Consumed: {totalCommEnergy} J \n \
                  - Total Training Time: {totalTrainTime} s \n \
                  - Total Number of Communications: {totalNumCommunications} \n")

            break

    print(f"\n=== FINAL RESULTS ===\n \
            - Total Computation Energy Consumed: {totalCompEnergy} J \n \
            - Total Communication Energy Consumed: {totalCommEnergy} J \n \
            - Total Training Time: {totalTrainTime} s \n \
            - Total Number of Communications: {totalNumCommunications} \n \
            - Final Accuracy: {accuracies}% \n \
            - Final Loss: {losses} \n")





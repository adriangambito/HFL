import numpy as np

from client_manager import ClientManager
from edge_server import EdgeServer

def hfl_clustering(config):

    print("\n\nCLUSTERING STARTED...")



    NUM_CLIENTS = config['NUM_CLIENTS']
    if NUM_CLIENTS <= 0:
        raise ValueError(f"Invalid number of clients: {NUM_CLIENTS}. Must be greater than 0.")
    clients = [ClientManager(id) for id in range(NUM_CLIENTS)]


    # Create edge servers (5 edge servers for 5 clusters)
    EDGE_SERVERS = config['EDGE_SERVERS']
    if EDGE_SERVERS <= 0:
        raise ValueError(f"Invalid number of edge servers: {EDGE_SERVERS}. Must be greater than 0.")
    edge_servers = [EdgeServer(server_id=i) for i in range(EDGE_SERVERS)]



    for i, client in enumerate(clients):
            edge_server = edge_servers[i % EDGE_SERVERS]
            edge_server.add_client(client)
            # print(f"Client {client.client_id} assigned to Edge Server {edge_server.get_server_id()}") 

    clustered_clients = {i: [] for i in range(EDGE_SERVERS)}

    for edge_server in edge_servers:
        edge_server.set_num_clients(edge_server.get_num_clients())
        clustered_clients[edge_server.get_server_id()] = edge_server.get_clients()
        # print(f"Edge Server {edge_server.get_server_id()} assigned {edge_server.get_num_clients()} clients.")
        # print(clustered_clients[edge_server.get_server_id()])

    cluster_statistics(config, clustered_clients)

    return clustered_clients





def cluster_statistics(config, clustered_clients, size_model_bits=7524672, cluster_local_iterations=None):
    print("\n\nCLUSTERING STATISTICS:\n")
    # if config['LOG_TO_FILE']:
    #     with open(config['LOG_FILE_PATH'], "a") as file:
    #         file.write("\nCLUSTERING STATISTICS:\n")

    for cluster_id, clients in clustered_clients.items():
        avg_energy = np.mean([client.energy_comp_sample for client in clients])
        avg_energy_comm = np.mean([client.computeEnergyCommunication(size_model_bits) for client in clients])
        avg_time = np.mean([client.train_time_sample for client in clients])
        cluster_size = len(clients)

        # with open(config['LOG_FILE_PATH'], "a") as file:
        #     file.write(f"Cluster {cluster_id}: Avg Energy Comp Sample: {avg_energy:.10f}, Avg Train Time Sample: {avg_time:.10f}, LOCAL_ITERATIONS: {cluster_local_iterations[cluster_id]}, clients: {cluster_size}\n")
        if cluster_local_iterations is not None:
            print(f"Cluster {cluster_id}: Avg Comp Energy: {avg_energy:.10f}, Avg Comm Energy: {avg_energy_comm:.10f}, Avg Time: {avg_time:.10f}, LOCAL_ITERATIONS: {cluster_local_iterations[cluster_id]}, clients: {cluster_size}\n")
        else:
            print(f"Cluster {cluster_id}: Avg Comp Energy: {avg_energy:.10f}, Avg Comm Energy: {avg_energy_comm:.10f}, Avg Time: {avg_time:.10f}, clients: {cluster_size}\n")
            # if config['LOG_TO_FILE']:
            #     with open(config['LOG_FILE_PATH'], "a") as file:
            #         file.write(f"Cluster {cluster_id}: Avg Comp Energy: {avg_energy:.10f}, Avg Comm Energy: {avg_energy_comm:.10f}, Avg Time: {avg_time:.10f}, clients: {cluster_size}\n")
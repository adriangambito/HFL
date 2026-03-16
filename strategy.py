import random

from flwr.server.strategy import FedAvg
from flwr.server.client_proxy import ClientProxy
from flwr.common import FitRes, FitIns, Parameters, parameters_to_ndarrays
from flwr.server.client_manager import ClientManager

class CustomFedAvg(FedAvg):

    def __init__(self, config, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.final_parameters = None
        self.total_samples = 0
        self.clientsMetricsTracker = {}

        # TODO: remove "config" parameter, is not used
        self.seed = config.get("SEED", 42)
        self.local_epochs = config.get("LOCAL_EPOCHS", 1)
    



    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> list[tuple[ClientProxy, FitIns]]:
        """Configure the next round of training."""

        config = {}
        if self.on_fit_config_fn is not None:
            config = self.on_fit_config_fn(server_round)
        fit_ins = FitIns(parameters, config)

        sample_size, _ = self.num_fit_clients(client_manager.num_available())
        client_manager.wait_for(sample_size)

        #random.seed(int(self.seed+server_round))
        available_cids = list(client_manager.all())
        sampled_cids = random.sample(available_cids, sample_size)
        clients = [client_manager.clients[cid] for cid in sampled_cids]

        return [(client, fit_ins) for client in clients]
    


    def aggregate_fit(self, 
                      server_round: int, 
                      results: list[tuple[ClientProxy, FitRes]], 
                      failures: list[tuple[ClientProxy, FitRes] | BaseException]
                      ) -> tuple[Parameters |None, dict[str, bool | bytes | float | int | str]]:
        
        print("\033[34mAGGREGATE_FIT\033[0m")

        results.sort(key=lambda x: x[1].metrics["client_id"])
        # random.seed(int(self.seed+server_round))
        # results.sort(key=lambda x: random.random())

        #print("CLIENTS IDS: ", [fit_res.metrics for _, fit_res in results])

        # * AGGREGATION OF PARAMETERS AND METRICS *
        ################################################################################
        parameters_aggregated, metrics_aggregated = super().aggregate_fit(server_round, results, failures)
        self.final_parameters = parameters_aggregated
        ################################################################################


        # * TOTAL SAMPLES COMPUTATION *
        ################################################################################
        total_samples = sum([fit_res.num_examples for _, fit_res in results])
        self.total_samples = total_samples
        print(f"Total samples aggregated in round {server_round}: {self.total_samples}")
        ################################################################################

        # *ENERGY CONSUMPTION AND TRAIN TIME COMPUTATION *
        ################################################################################
        clientsMetrics = [fit_res.metrics for _, fit_res in results]

        for client in clientsMetrics:
            client_id = client["client_id"]

            if client_id not in self.clientsMetricsTracker:
                self.clientsMetricsTracker[client_id] = client
            else:
                self.clientsMetricsTracker[client_id]['consumedEnergyComputation'] += client['consumedEnergyComputation']
                self.clientsMetricsTracker[client_id]['consumedEnergyCommunication'] += client['consumedEnergyCommunication']
                self.clientsMetricsTracker[client_id]['trainTimeComputation'] += client['trainTimeComputation']
                self.clientsMetricsTracker[client_id]['numCommunications'] += client['numCommunications']
        ################################################################################


        return parameters_aggregated, metrics_aggregated
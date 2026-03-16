import random

SEED = 42
random.seed(SEED)

class ClientManager():
    def __init__(self, id):
        self.client_id = id
        self.battery_capacity_wh = 98.8       # Watt-hours
        self.battery_capacity_joules = 355680.0  # Convert Wh to Joules (98.8 * 3600)
        self.actual_battery_capacity_J = 355680.0
        self.actual_batteryLevel_percentage = 100.0
        self.num_communications = 0
        self.included = True

        # Energy and training time per sample (example: MNIST) {ORIGINAL}
        self.energy_comp_sample = 0.000012544  # Joules per sample computation
        self.train_time_sample = 0.00012544    # Seconds per sample
        self.channel_capacity = 5.672e6  # bits/s (from Shannon-Hartley theorem)
        self.transmitter_power = 0.5     # Watts

        #self.energy_comp_sample = random.uniform(0.00001, 0.00002)  # Joules per sample computation
        #self.train_time_sample = random.uniform(0.0001, 0.0002)     # Seconds per sample

        # Energy consumption per sample
        # min_energy, max_energy = 0.00001, 0.00004
        # self.energy_comp_sample = random.uniform(0.00001, 0.00004)

        # Computation time per sample
        # min_time, max_time = 0.0001, 0.0004
        # self.train_time_sample = min_time + (self.energy_comp_sample - min_energy) * (max_time - min_time) / (max_energy - min_energy)

        # Channel capacity (es. 2 Mbps – 8 Mbps)
        # min_channel, max_channel = 2e6, 8e6  # in bit/s
        # self.channel_capacity = random.uniform(min_channel, max_channel)
        #self.channel_capacity = 5.672e6
        
        # Transmitter power (es. 0.3W – 0.7W)
        # min_power, max_power = 0.3, 0.65  # in Watt
        # self.transmitter_power = random.uniform(min_power, max_power)
        #self.transmitter_power = 0.5
        

        # Track total energy consumption
        self.total_comp_energy = 0.0
        self.total_comm_energy = 0.0


    def __repr__(self):
        return (f"ClientManager(client_id={self.client_id}, "
                f"energy_comp_sample={self.energy_comp_sample:.8f} J/sample) "
                f"train_time_sample={self.train_time_sample:.6f} s, "
                f"channel_capacity={self.channel_capacity:.2e} bits/s, "
                f"transmitter_power={self.transmitter_power:.2f} W"
                )


    def getEnergyLevel(self):
        return self.actual_batteryLevel_percentage


    def getEnergyCapacity(self):
        return self.actual_battery_capacity_J


    def computeEnergyComputation(self, num_samples):
        """
        Compute the energy consumed for the computation of given number of samples.
        Returns the energy consumed (Joules).
        """
        # energyCompConsumed = round((num_samples * self.energy_comp_sample), 10)
        energyCompConsumed = 0.0024
        self.total_comp_energy += energyCompConsumed  # Update total computation energy
        return energyCompConsumed


    def computeEnergyCommunication(self, size_model_bits):
        """
        Compute the energy consumed for transmitting the local model updates.
        Returns the energy consumed (Joules).
        """
        communicationLatency = round((size_model_bits / self.channel_capacity), 10)
        # energyCommConsumed = round((self.transmitter_power * communicationLatency), 10)
        energyCommConsumed = 0.0616
        self.total_comm_energy += energyCommConsumed  # Update total communication energy
        return energyCommConsumed


    def computeTrainTimeComputation(self, num_samples):
        """
        Compute the training time for a given number of samples.
        Returns the time in seconds.
        """
        training_time = round((num_samples * self.train_time_sample), 10)
        return training_time


    def getNumCommunications(self):
        return self.num_communications


    def setNumCommunications(self, num_communications):
        self.num_communications = num_communications


    def decreaseEnergyLevel(self, amountEnergyConsumed):
        """
        Decrease the battery level by a certain amount.
        """
        remainingBatteryCapacity = round((self.actual_battery_capacity_J - amountEnergyConsumed), 10)
        self.actual_battery_capacity_J = remainingBatteryCapacity
        energyLevelPercentage = round(((self.actual_battery_capacity_J * 100) / self.battery_capacity_joules), 10)
        self.actual_batteryLevel_percentage = energyLevelPercentage


    def decreaseEnergyLevelCommunication(self, updatedParamsBits):
        """
        Decrease the battery level by energy consumed for communication.
        """
        communicationLatency = round((updatedParamsBits / self.channel_capacity), 5)
        commEnergyConsumed = self.transmitter_power * communicationLatency
        self.total_comm_energy += commEnergyConsumed  # Track communication energy
        remainingBatteryCapacity = self.actual_battery_capacity_J - commEnergyConsumed
        self.actual_battery_capacity_J = remainingBatteryCapacity

        self.num_communications += 1
        energyLevelPercentage = round(((self.actual_battery_capacity_J * 100) / self.battery_capacity_joules), 2)
        self.actual_batteryLevel_percentage = energyLevelPercentage


    def getTotalConsumedComputationalEnergy(self):
        """
        Return the total computational energy consumed by this device (Joules).
        """
        return self.total_comp_energy


    def getTotalConsumedCommunicationEnergy(self):
        """
        Return the total communication energy consumed by this device (Joules).
        """
        return self.total_comm_energy

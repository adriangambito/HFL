import os

import torch
import numpy as np
import random

from model import Net, get_parameters

def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.use_deterministic_algorithms(True)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False




def get_size_model_bits(model) -> int:
    total_bits = sum(p.numel() * p.element_size() * 8 for p in model.parameters())
    return total_bits



def initialization_model(config):
    if config['DATASET'] == "mnist" or config['DATASET'] == "fashion-mnist":
        # Initialize the model for MNIST or Fashion-MNIST
        model = Net()
    elif config['DATASET'] == "cifar10":
        # Initialize the model for CIFAR10
        print("Initializing ResNet18 for CIFAR10")
        model = Net()
    else:
        raise ValueError(f"UNSUPPORTED DATASET: {config['DATASET']}")

    # Nome file inizializzazione
    init_params_file = "Models/init/%s_init_params.pth" % config['DATASET']

    if os.path.exists(init_params_file):
        print(f"INIT PARAMETERS FOUND: {init_params_file}.\nUPLOADING PARAMETERS...")
        model.load_state_dict(torch.load(init_params_file))
    else:
        print(f"NO INIT PARAMETERS FILE {init_params_file} FOUND.\nINITIALIZING NEW MODEL AND SAVE PARAMETERS...")
        torch.save(model.state_dict(), init_params_file)

    # Ora calcolo la dimensione del modello
    model_size_bits = get_size_model_bits(model)
    print(f"Model size in bits: {model_size_bits}")

    init_params = get_parameters(model)

    return model, init_params
import os
import random
import numpy as np

os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import torch

def setup_deterministic():
    """Configure PyTorch for deterministic behavior"""
    seed = 68
    
    random.seed(seed)
    np.random.seed(seed)
    
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # Enable deterministic algorithms
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    
    print("PyTorch configured for deterministic execution")

setup_deterministic()
# -*- coding: utf-8 -*-
import os
from dotenv import load_dotenv

load_dotenv()

class RuntimeConfig:
    PSICHIC_PATH = os.path.dirname(os.path.abspath(__file__))
    device = os.environ.get("DEVICE_OVERRIDE")
    DEVICE = ["cpu" if device=="cpu" else "cuda:0"][0]
    MODEL_PATH = os.path.join(PSICHIC_PATH, 'trained_weights', 'TREAT2')
    BATCH_SIZE = 128
    

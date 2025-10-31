import torch
import torch.nn as nn
from modules import TimeEncoder,Attention

class TGAT(nn.Module):
    def __init__(self): 
        super().__init__()


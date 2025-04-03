from random import sample
from numpy import save
from test.throughout_test import throughout_test
from ultils.GCB import gcb_rebalance
from ultils.linear_programming import balance_channels
from ultils.draw import draw
from data_processing.data_preprocessing import pross_data
from data_processing.fix import save_data, load_saved_data
from models.balance_model import Transformer_PolicyNet
import json
import torch
import pickle
import copy

save_data(dataset='ripple', sample=1500)
state, bit_values = load_saved_data(dataset='ripple')
state, bit_values = load_saved_data(dataset='lighting')
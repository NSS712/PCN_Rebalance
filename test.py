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

# save_data()
state, bit_values = load_saved_data()
config = json.load(open('config/DRLPCR.json', 'r'))


tpcr = Transformer_PolicyNet(config)
tpcr.load_state_dict(torch.load('saved_model/ep_100_policy_network.pth'))
tpcr.eval()
tpcr.to('cuda')

# t1 = throughout_test(state, bit_values, balance_method=gcb_rebalance)
# with open('result/gcb.pkl', 'wb') as f:
#     pickle.dump(t1, f)
with open('result/gcb.pkl', 'rb') as f:
    t1 = pickle.load(f)

# t2 = throughout_test(state, bit_values, balance_method=balance_channels)
# with open('result/ling.pkl', 'wb') as f:
#     pickle.dump(t2, f)
with open('result/ling.pkl', 'rb') as f:
    t2 = pickle.load(f)

# t3 = throughout_test(state, bit_values, balance_method=None)
# with open('result/no_balance.pkl', 'wb') as f:
#     pickle.dump(t3, f)
with open('result/no_balance.pkl', 'rb') as f:
    t3 = pickle.load(f)


step = 1
t4 = throughout_test(copy.copy(state), bit_values, balance_method=tpcr.caculate_next_state)
while t4[0][-1] < 2400:
    step += 1
    print("step:", step)
    t4 = throughout_test(copy.copy(state), bit_values, balance_method=tpcr.caculate_next_state)
with open('result/DRLPCR.pkl', 'wb') as f:
    pickle.dump(t4,f)
# with open('result/DRLPCR.pkl', 'rb') as f:
#     t4 = pickle.load(f)

draw(t1, t2, t3, t4)

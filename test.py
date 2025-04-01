from numpy import save
from test.throughout_test import throughout_test
from ultils.GCB import gcb_rebalance
from ultils.linear_programming import balance_channels
from ultils.draw import draw
from data_processing.data_preprocessing import pross_data
from data_processing.fix import save_data, load_saved_data

# save_data()
state, bit_values = load_saved_data()
# t1 = throughout_test(state, bit_values, balance_method=gcb_rebalance)
# t2 = throughout_test(state, bit_values, balance_method=balance_channels)
# t3 = throughout_test(state, bit_values, balance_method=None)
# draw(t1, t2, t3)

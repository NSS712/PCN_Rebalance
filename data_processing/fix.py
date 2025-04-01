from ultils.entity import *
from data_processing.data_preprocessing import pross_data
import pickle


def save_data():
    state, bit_values = pross_data(sample=200, need_path=True)
    with open('data/processed/data.pkl', 'wb') as f:
        pickle.dump(state, f)
        pickle.dump(bit_values, f)

def load_saved_data():
    with open('data/processed/data.pkl', 'rb') as f:
        state = pickle.load(f)
        bit_values = pickle.load(f)
    print(f"数据加载完成, {state.node_num}个节点， {len(state.channels)}个通道")
    bit_values = [x//200 for x in bit_values]
    return state, bit_values


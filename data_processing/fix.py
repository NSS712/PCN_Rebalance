from ultils.entity import *
from data_processing.data_preprocessing import pross_data
import pickle


def save_data(dataset='lightning', sample=200):
    state, bit_values = pross_data(sample=sample, need_path=True, dataset=dataset)
    if dataset == "ripple":
        with open('data/processed/ripple_data.pkl', 'wb') as f:
            pickle.dump(state, f)
            pickle.dump(bit_values, f)
    elif dataset == "lightning":
        with open('data/processed/data.pkl', 'wb') as f:
            pickle.dump(state, f)
            pickle.dump(bit_values, f)
    else:
        raise NotImplementedError

def load_saved_data(dataset='lighting'):
    if dataset == "lighting":
        with open('data/processed/data.pkl', 'rb') as f:
            state = pickle.load(f)
            bit_values = pickle.load(f)
            bit_values = [x//200 for x in bit_values]
    elif dataset == "ripple":
        with open('data/processed/ripple_data.pkl', 'rb') as f:
            state = pickle.load(f)
            bit_values = pickle.load(f)
            bit_values = [x//20 for x in bit_values]
    else:
        raise NotImplementedError
    print(f"数据加载完成, {state.node_num}个节点， {len(state.channels)}个通道")
    return state, bit_values


from numpy import save
from data_processing.data_preprocessing import pross_data
import json
from models.trainer import DRLPCRTrainer
import torch
from data_processing.fix import save_data, load_saved_data
from test.throughout_test import throughout_test
import copy


if __name__ == "__main__":
    # save_data()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    config = json.load(open('config/DRLPCR.json'))
    initial_state, amounts = load_saved_data()
    trainer = DRLPCRTrainer(initial_state, amounts, config, net="TPCR")
    throughout_test(copy.copy(initial_state), amounts,balance_method=None)
    for eposide in range(config['eposide_num']):
        trainer.train_eposide()
        throughout_test(copy.copy(initial_state), amounts, balance_method=trainer.policy_network.caculate_next_state)




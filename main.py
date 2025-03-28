from data_processing.data_preprocessing import pross_data
import json
from models.trainer import DRLPCRTrainer
import torch


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    config = json.load(open('config/DRLPCR.json'))
    initial_state, amounts = pross_data()
    trainer = DRLPCRTrainer(initial_state, amounts, config)

    for eposide in range(config['eposide_num']):
        trainer.train_eposide()
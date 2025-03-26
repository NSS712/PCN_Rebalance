from data_processing.data_preprocessing import pross_data
import json
from models.trainer import DRLPCRTrainer


if __name__ == "__main__":
    config = json.load(open('config/DRLPCR.json'))
    initial_state, amounts = pross_data()
    trainer = DRLPCRTrainer(initial_state, amounts, config)

    for eposide in range(config['eposide_num']):
        trainer.train_eposide()
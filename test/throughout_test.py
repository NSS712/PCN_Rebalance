from numpy.random import f
from ultils.linear_programming import balance_channels
from data_processing.data_preprocessing import pross_data
from ultils.entity import *
import random
from tqdm import tqdm

def throughout_test(state, bit_values, balance_method=balance_channels):
    target_transaction_num = 10000
    transaction_num = success_num = throughout = 0
    reset_num = 0
    balance_num = 0
    success_res = []
    throughout_res = []
    for i in tqdm(range(target_transaction_num), leave=False):
        amount = random.choice(bit_values)
        status = state.random_transaction(amount)
        if status:
            success_num += 1
            throughout += amount
        else:
            reset_num += 1
        
        if reset_num == 15:
            balance_num += 1
            reset_num = 0
            pre_index =state.compute_balance_index()
            if balance_method:
                state = balance_method(state)
            end_index = state.compute_balance_index()
            # print(f"当前交易次数: {transaction_num},第{balance_num}次调整,调整前平衡指数: {pre_index}, 调整后平衡指数: {end_index}")
            
        transaction_num += 1
        # if transaction_num % 1000 == 0:
        #     print(f"当前交易次数: {transaction_num}, 成功次数: {success_num}, 吞吐量: {throughout}")
        success_res.append(success_num)
        throughout_res.append(throughout)
    print("\033[2K\033[G")
    print(f"成功率: {success_num / transaction_num}")
    return success_res, throughout_res
 
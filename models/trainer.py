import copy
from math import gamma 
import torch
from ultils.entity import State, Channel, Path
from data_processing.data_preprocessing import pross_data
from models.model import PolicyNet, ValueNet, EmbeddingNet
import numpy as np
import pickle
import random
# 定义训练和更新的类
class DRLPCRTrainer:
    def __init__(self, initial_state, amounts, config):
        self.config = config
        self.amounts = amounts
        self.state = initial_state
        embedding_net = EmbeddingNet(config, initial_state)
        self.policy_network = PolicyNet(config, embedding_net)
        self.value_network = ValueNet(config, embedding_net)
        self.simulator = Simulator(self.config, self.amounts)
        self.optimizer = torch.optim.Adam(list(self.policy_network.parameters()) + list(self.value_network.parameters()), lr=self.config['learning_rate'])
        self.eposide_num = 0
        self.step = 0
        self.losses = {}

    def train_eposide(self):
        """
        进行一次模拟、采样和更新的过程
        """
        batch_size = self.config['batch_size']
        trajectory = self.simulator.generate_trajectory(self.state, self.policy_network)
        for step in range(self.config['buffer_size'] // batch_size):
            batch = {'states': trajectory['states'][step*batch_size:(step+1)*batch_size], 'actions': trajectory['actions'][step*batch_size:(step+1)*batch_size], 'rewards': trajectory['rewards'][step*batch_size:(step+1)*batch_size]}
            self.train_step(batch)
        self.save_data(trajectory)
        self.eposide_num += 1
        
    def save_data(self, trajectory):
        "保存模型"
        torch.save(self.policy_network.state_dict(), 'saved_model/ep_{}/policy_network.pth'.format(self.eposide_num))
        torch.save(self.value_network.state_dict(), 'saved_model/ep_{}/value_network.pth'.format(self.eposide_num))
        with open('data/trajectory/{}.plk'.format(self.eposide_num), 'wb') as f:
            pickle.dump(trajectory, f)
                
        
    def train_step(self, batch_data):
        """
        一次训练步骤，进行梯度更新
        batch_data['states'] # (batch, t)
        batch_data['actions'] # (batch, T, 3, P)
        batch_data['rewards'] # (batch, T)
        """
        c1 = self.config['c1']
        c2 = self.config['c2']
        gamma = self.config['gamma']
        epsilon = self.config['epsilon']

        states = batch_data['states'] # (batch, t)
        behavior_outputs = batch_data['actions'] # (batch, T, 3, P)
        behavior_rewards = batch_data['rewards'] # (batch, T)

        # 把states的二维list变成（t,batch）
        states = list(map(list, zip(*states))) # (t, batch)
        policy_outputs = []
        behavior_values = []
        for t_batch in states: # (batch)
            policy_step_outputs = self.policy_network.caculate(t_batch)  # (batch, 3, p), p是path的个数, 3是 mean, std, action
            policy_outputs.append(policy_step_outputs)
            behavior_step_values = self.value_network.caculate(t_batch)  # (batch, 1)
            behavior_values.append(behavior_step_values)

        behavior_values = torch.stack(behavior_values, dim=1) # (batch, t)
        # 计算策略损失、价值损失和熵项
        policy_loss = self.calculate_policy_loss(policy_outputs, behavior_outputs, behavior_rewards, behavior_values)
        value_loss = self.calculate_value_loss(behavior_values, behavior_rewards)
        entropy = self.calculate_policy_entropy(policy_outputs)
        # 计算总损失
        total_loss = policy_loss + c1 * value_loss - c2 * entropy

        # 反向传播和梯度更新
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        self.step += 1
        print("eposide: {}, step: {}, loss: {}, policy_loss: {}, value_loss: {}, entropy: {}".format(self.eposide_num, self.step, total_loss, policy_loss, value_loss, entropy))
        self.losses[self.step] = (total_loss, policy_loss, value_loss, entropy)

    def calculate_policy_loss(self, policy_outputs, behavior_outputs, behavior_rewards, behavior_values):
        """
        policy_outputs, value_outputs, actions_all, rewards
        计算策略损失
        :param policy_outputs: 策略网络的输出（动作的概率分布）
        :param actions: 实际采取的动作
        :param discounted_rewards: 折扣累积奖励
        :param value_estimates: 价值网络的输出（状态的价值估计）
        :return: 策略损失
        """
        gamma = self.config['gamma']
        J = self.config['J']
        epsilon = self.config['epsilon']  #剪辑操作的参数

        policy_mean, policy_std, policy_action = policy_outputs
        behavior_mean, behavior_std, behavior_action = behavior_outputs
        # 计算概率比
        ratio = self.calculate_probability_ratio(policy_mean, policy_std, behavior_mean, behavior_std, behavior_action)
        # 计算优势函数
        advantage = self.calculate_advantage(behavior_rewards, behavior_values)
        L_cpi = ratio * advantage
        L_clip = torch.clamp(ratio, 1 - epsilon, 1 + epsilon) * advantage
        policy_loss = -torch.mean(torch.min(L_cpi, L_clip))
        return policy_loss

    def calculate_probability_ratio(self, mean_theta, std_theta, mean_behavior, std_behavior, action):
        """
        此函数用于计算概率比
        :param mean_theta: 当前策略网络的均值    (batch, t, p)
        :param std_theta: 当前策略网络的标准差    (batch, t, p)
        :param mean_behavior: 行为策略网络的均值  (batch, t, p)
        :param std_behavior: 行为策略网络的标准差  (batch, t, p)
        :param action: behavior的经验动作  (batch, t, p)
        :return: 概率比
        """
        # 计算当前策略网络的概率密度
        prob_theta = (1 / (np.sqrt(2 * np.pi) * std_theta)) * np.exp(-((action - mean_theta) ** 2) / (2 * std_theta ** 2))
        # 计算行为策略网络的概率密度
        prob_behavior = (1 / (np.sqrt(2 * np.pi) * std_behavior)) * np.exp(-((action - mean_behavior) ** 2) / (2 * std_behavior ** 2))
        # 计算概率比
        ratio = prob_theta / prob_behavior
        return ratio

    def calculate_advantage(self, rewards, values):
        """
        此函数用于计算优势函数 A_t,输入为 (batch, T) 的张量
        :param rewards: 形状为 (batch, T) 的张量，包含每个批次每个时间步的奖励
        :param values: 形状为 (batch, T) 的张量，包含每个批次每个时间步的状态价值
        :return: 形状为 (batch, T) 的张量，包含每个批次每个时间步的优势函数值
        """
        gamma = self.config['gamma'] #折扣因子
        J = self.config['J'] #优势函数向前看的步数
        batch_size, T = rewards.shape
        advantages = torch.zeros((batch_size, T), dtype=torch.float32)

        for t in range(T):
            for j in range(min(J, T - t)):
                if t + j + 1 < T:
                    delta = rewards[:, t + j] + gamma * values[:, t + j + 1] - values[:, t + j]
                else:
                    delta = rewards[:, t + j] - values[:, t + j]
                advantages[:, t] += (gamma ** j) * delta

        return advantages

    def calculate_value_loss(self, value_outputs, rewards):
        """
        计算价值损失
        :param value_outputs: 价值网络的输出（状态的价值估计）, (batch, T)
        :param discounted_rewards: 每个中间状态的奖励， (batch, T)
        :param gamma: 折扣因子
        :return: 价值损失
        """
        gamma = self.config['gamma']
        batch_size, T = rewards.shape
        discounted = torch.zeros_like(rewards)
        for b in range(batch_size):
            running_add = 0
            for t in reversed(range(T)):
                running_add = running_add * gamma + rewards[b, t]
                discounted[b, t] = running_add
        
        value_loss = torch.mean(torch.sum((discounted - value_outputs) ** 2), dim=(1,2))
        return value_loss

    def calculate_policy_entropy(self, policy_outputs):
        """
        此函数用于计算策略 π_θ 的熵
        :param policy_outputs: 形状为 (batch_size, steps, 3, p) 的数组
        :return: 策略 π_θ 的熵
        """
        batch_size, steps, _ , p= policy_outputs.shape
        stds = policy_outputs[:, :, 1, :]  # 提取标准差
        entropy_per_step = 0.5 * np.log(2 * np.pi * np.e * stds ** 2)
        total_entropy = np.sum(entropy_per_step)
        return -total_entropy / (batch_size * steps * p)

class Simulator:
    def __init__(self, config, amounts):
        self.config = config
        self.amounts = amounts
        self.transactions_num = 0
        self.failed_transactions_num = 0

    def generate_trajectory(self, state, policy_network):
        "生成交易信息"
        self.state = state
        self.policy_network = policy_network
        trajectory = {'states': [], 'actions': [], 'rewards': []}
        while len(trajectory) < self.config['buffer_size']:
            t = 0
            while t < self.config['trigger_threshold']:
                amount = random.choice(self.amounts)
                if not self.state.random_transaction(amount):
                    t += 1   
                    self.failed_transactions_num += 1
                self.transactions_num += 1
                if self.transactions_num % 1000 == 0:
                    print("transactions_num: {}, failed_transactions_num: {}, failed rate:{:.2f}".format(self.transactions_num, self.failed_transactions_num, self.failed_transactions_num / self.transactions_num ))
            states, actions, rewards = self.policy_network.caculate_T_steps([self.state])
            trajectory['states'].extend(states[0][:-1])
            trajectory['actions'].extend(actions)
            trajectory['rewards'].extend(rewards)
            self.state = states[0][-1]
        return trajectory
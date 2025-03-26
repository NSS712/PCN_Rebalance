import copy 
import torch
from ultils.entity import State, Channel, Path
from data_processing.data_preprocessing import pross_data
from models.model import PolicyNet, ValueNet
import numpy as np
import pickle
import random
# 定义训练和更新的类
class DRLPCRTrainer:
    def __init__(self, initial_state, amounts, config):
        self.config = config
        self.amounts = amounts
        self.state = initial_state
        self.policy_network = PolicyNet(config)
        self.value_network = ValueNet(config)
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
        for step in range(self.config['buffer_size'] / batch_size):
            batch = trajectory[step * batch_size:(step + 1) * batch_size]  # (batch, T, (state, action, reward))
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
        batch_data: (batch, T, (state, action))
        """
        c1 = self.config['c1']
        c2 = self.config['c2']
        gamma = self.config['gamma']
        epsilon = self.config['epsilon']

        states = [[row[0] for row in T] for T in batch_data]
        actions = [[row[1] for row in T] for T in batch_data]
        rewards = [[row[0].compute_reward() for row in T] for T in batch_data]

        # 前向传播，计算策略网络的输出
        policy_outputs = self.policy_network(states)  # (batch, T, (mean, std)) ,(batch, T, action)
        for _ in range(self.config['trigger_threshold']):
            action = self.policy_network(self.state)
            new_state = state.act(action)
            reward = new_state.compute_reward()
            states.append(new_state)
            actions.append(action)
            state = new_state
            rewards.append(reward)

        # 计算折扣累积奖励
        discounted_rewards = self.calculate_discounted_rewards(rewards, gamma)
        # 前向传播，计算价值网络的输出
        value_estimates = []
        for state in states:
            value_estimate = self.value_network(state)
            value_estimates.append(value_estimate)

        # 计算策略损失、价值损失和熵项
        policy_loss = self.calculate_policy_loss(policy_outputs, actions, discounted_rewards, value_estimates, epsilon)
        value_loss = self.calculate_value_loss(value_estimates, discounted_rewards)
        entropy = self.calculate_entropy(policy_outputs)

        # 计算总损失
        total_loss = policy_loss + c1 * value_loss - c2 * entropy

        # 反向传播和梯度更新
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        self.step += 1
        print("eposide: {}, step: {}, loss: {}, policy_loss: {}, value_loss: {}, entropy: {}".format(self.eposide_num, self.step, total_loss, policy_loss, value_loss, entropy))
        self.losses[self.step] = (total_loss, policy_loss, value_loss, entropy)

        """
        计算折扣累积奖励
        :param rewards: 奖励序列
        :param gamma: 折扣因子
        :return: 折扣累积奖励序列
        """
        # 这里实现折扣累积奖励的计算，暂时省略具体实现
        discounted_rewards = None
        return discounted_rewards

    def calculate_totol_loss(self, policy_outputs, value_outputs, actions_all, rewards, gamma, epsilon, J, c1, c2):
        "计算总损失"
        policy_loss = self.calculate_policy_loss(policy_outputs, value_outputs, actions_all, rewards, gamma, J, epsilon)
        value_loss = self.calculate_value_loss(value_outputs, rewards, gamma)
        entropy = self.calculate_policy_entropy(policy_outputs)
        # 计算总损失
        total_loss = policy_loss + c1 * value_loss - c2 * entropy
        return total_loss

    def calculate_policy_loss(self, policy_outputs, value_outputs, actions_all, rewards, gamma, J, epsilon):
        """
        计算策略损失
        :param policy_outputs: 策略网络的输出（动作的概率分布）
        :param actions: 实际采取的动作
        :param discounted_rewards: 折扣累积奖励
        :param value_estimates: 价值网络的输出（状态的价值估计）
        :param epsilon: 剪辑操作的参数
        :return: 策略损失
        """
        (new_mean, new_std, new_action) = policy_outputs
        (mean, std, actions) = actions_all
        # 计算概率比
        ratio = self.calculate_probability_ratio(new_mean, new_std, mean, std, actions)
        # 计算优势函数
        advantage = self.calculate_advantage(rewards, value_outputs, gamma, J)
        L_cpi = ratio * advantage
        L_clip = torch.clamp(ratio, 1 - epsilon, 1 + epsilon) * advantage
        policy_loss = -torch.mean(torch.min(L_cpi, L_clip))
        return policy_loss

    def calculate_probability_ratio(self, mean_theta, std_theta, mean_behavior, std_behavior, action):
        """
        此函数用于计算概率比
        :param mean_theta: 当前策略网络的均值
        :param std_theta: 当前策略网络的标准差
        :param mean_behavior: 行为策略网络的均值
        :param std_behavior: 行为策略网络的标准差
        :param action: 采样得到的动作
        :return: 概率比
        """
        # 计算当前策略网络的概率密度
        prob_theta = (1 / (np.sqrt(2 * np.pi) * std_theta)) * np.exp(-((action - mean_theta) ** 2) / (2 * std_theta ** 2))
        # 计算行为策略网络的概率密度
        prob_behavior = (1 / (np.sqrt(2 * np.pi) * std_behavior)) * np.exp(-((action - mean_behavior) ** 2) / (2 * std_behavior ** 2))
        # 计算概率比
        ratio = prob_theta / prob_behavior
        return ratio

    def calculate_advantage(self, rewards, values, gamma, J):
        """
        此函数用于计算优势函数 A_t
        :param rewards: 奖励列表，包含每个时间步的奖励
        :param values: 状态价值估计列表，包含每个时间步的状态价值
        :param gamma: 折扣因子
        :param J: 优势函数向前看的步数
        :return: 优势函数值列表，包含每个时间步的优势函数值
        """
        num_steps = len(rewards)
        advantages = np.zeros(num_steps)

        for t in range(num_steps):
            advantage = 0
            for j in range(min(J, num_steps - t)):
                if t + j + 1 < num_steps:
                    delta = rewards[t + j] + gamma * values[t + j + 1] - values[t + j]
                else:
                    delta = rewards[t + j] - values[t + j]
                advantage += (gamma ** j) * delta
            advantages[t] = advantage

        return advantages

    def calculate_value_loss(self, value_outputs, rewards, gamma):
        """
        计算价值损失
        :param value_outputs: 价值网络的输出（状态的价值估计）, (batch, T, 1)
        :param discounted_rewards: 每个中间状态的奖励， (batch, T, 1)
        :param gamma: 折扣因子
        :return: 价值损失
        """
        batch_size, T, _ = rewards.shape
        discounted = torch.zeros_like(rewards)
        for b in range(batch_size):
            running_add = 0
            for t in reversed(range(T)):
                running_add = running_add * gamma + rewards[b, t, 0]
                discounted[b, t, 0] = running_add
        
        value_loss = torch.mean(torch.sum((discounted - value_outputs) ** 2), dim=(1,2))
        return value_loss

    def calculate_policy_entropy(self, policy_outputs):
        """
        此函数用于计算策略 π_θ 的熵
        :param policy_outputs: 形状为 (batch_size, steps, 3) 的数组，每个元素包含 (mean, std, action)
        :return: 策略 π_θ 的熵
        """
        batch_size, steps, _ = policy_outputs.shape
        stds = policy_outputs[:, :, 1]  # 提取标准差
        entropy_per_step = 0.5 * np.log(2 * np.pi * np.e * stds ** 2)
        total_entropy = np.sum(entropy_per_step)
        return -total_entropy / (batch_size * steps)

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
        trajectory = []
        while len(trajectory) < self.config['buffer_size']:
            t = 0
            while t < self.config['trigger_threshold']:
                amount = random.choice(self.amounts)
                if not self.state.random_transaction(amount):
                    t += 1   
                    self.failed_transactions_num += 1
                self.transactions_num += 1
                if self.transactions_num % 1000 == 0:
                    print("transactions_num: {}, failed_transactions_num: {}, failed rate:{}".format(self.transactions_num, self.failed_transactions_num, self.failed_transactions_num / self.transactions_num ))
            action = self.policy_network.caculate([self.state])
            trajectory.append((copy.copy(self.state), action))
            self.state.act(action[2][0])
        return trajectory
    
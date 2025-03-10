import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.distributions import Categorical

class ActorNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(ActorNetwork, self).__init__()
        self.actor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Softmax(dim=-1)
        )
    
    def forward(self, state):
        return self.actor(state)

class CriticNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(CriticNetwork, self).__init__()
        self.critic = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, state):
        return self.critic(state)

class ActorCritic:
    def __init__(self, input_dim, hidden_dim, output_dim, learning_rate=3e-4, gamma=0.99):
        self.actor = ActorNetwork(input_dim, hidden_dim, output_dim)
        self.critic = CriticNetwork(input_dim, hidden_dim)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=learning_rate)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=learning_rate)
        self.gamma = gamma
        
    def select_action(self, state):
        state = torch.FloatTensor(state)
        action_probs = self.actor(state)
        dist = Categorical(action_probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action)
    
    def update(self, state, action, reward, next_state, done):
        # 转换为tensor
        state = torch.FloatTensor(state)
        next_state = torch.FloatTensor(next_state)
        
        # 计算当前状态的价值和下一状态的价值
        value = self.critic(state)
        next_value = self.critic(next_state)
        
        # 计算TD误差
        if done:
            advantage = reward - value
        else:
            advantage = reward + self.gamma * next_value.detach() - value
            
        # 计算Actor的损失
        action_probs = self.actor(state)
        dist = Categorical(action_probs)
        log_prob = dist.log_prob(torch.tensor(action))
        actor_loss = -log_prob * advantage.detach()
        
        # 计算Critic的损失
        critic_loss = advantage.pow(2)
        
        # 更新网络
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        return actor_loss.item(), critic_loss.item()

# 使用示例
if __name__ == "__main__":
    # 环境参数示例
    input_dim = 4  # 状态空间维度
    hidden_dim = 128  # 隐藏层维度
    output_dim = 2  # 动作空间维度
    
    # 初始化 Actor-Critic
    ac = ActorCritic(input_dim, hidden_dim, output_dim)
    
    # 训练循环示例
    num_episodes = 10
    for episode in range(num_episodes):
        state = np.random.rand(input_dim)  # 示例状态
        done = False
        total_reward = 0
        
        while not done:
            # 选择动作
            action, _ = ac.select_action(state)
            
            # 与环境交互（这里需要实际环境）
            next_state = np.random.rand(input_dim)  # 示例下一状态
            reward = np.random.rand()  # 示例奖励
            done = np.random.rand() > 0.95  # 示例终止条件
            
            # 更新网络
            actor_loss, critic_loss = ac.update(state, action, reward, next_state, done)
            
            total_reward += reward
            state = next_state
            
        if episode:
            print(f"Episode {episode}, Total Reward: {total_reward:.2f}") 
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random

# CNN模型
class DQNModel(nn.Module):

    def __init__(self, n_action=2):
        super(DQNModel, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=4, out_channels=32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)
        self.fc = nn.Linear(7 * 7 * 64, 512)
        self.out = nn.Linear(512, n_action)
        self._create_weights()

    # 初始化权重和偏置
    def _create_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.uniform_(m.weight, -0.01, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc(x))
        output = self.out(x)
        return output

# 具有经验回放的 DQN
class DQN():
    def __init__(self, n_action, lr=1e-6):
        # 设置损失函数为均方误差损失
        self.criterion = torch.nn.MSELoss()
        self.model = DQNModel(n_action)
        # 判断是否有可用的gpu，如果有就把模型和损失函数都放到gpu上
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.criterion.to(self.device)
        # # 创建一个Adam优化器，用来更新模型的参数，参数为模型的参数和学习率
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr)

    # 根据输入状态估计输出 Q 值
    def predict(self, s):
        # 把输入的状态也放到gpu上
        s = s.to(self.device)
        # 调用模型的前向传播方法，返回每个动作的预测值
        return self.model(torch.Tensor(s))

    # 给定训练样本更新模型的权重
    def update(self, y_predict, y_target):
        # 计算预测值和目标值之间的均方误差损失
        loss = self.criterion(y_predict, y_target)
        # 清空优化器的梯度
        self.optimizer.zero_grad()
        # 反向传播计算梯度
        loss.backward()
        # 优化器更新模型的参数
        self.optimizer.step()
        return loss.item()

    # 从记忆中采样一批数据，进行经验回放
    def replay(self, memory, replay_size, gamma):
        # 如果记忆的长度大于等于采样的大小
        if len(memory) >= replay_size:
            # 从记忆中随机采样一批数据
            replay_data = random.sample(memory, replay_size)
            # 将采样的数据分别解压为状态，动作，下一个状态，奖励和是否结束的元组
            state_batch, action_batch, next_state_batch, reward_batch, done_batch = zip(*replay_data)
            # 拼接状态元组
            state_batch = torch.cat(tuple(state for state in state_batch))
            # 拼接下一个状态元组
            next_state_batch = torch.cat(tuple(state for state in next_state_batch))
            # 调用predict方法，根据状态和下一个状态的tensor，得到对应的Qvalue
            q_values_batch = self.predict(state_batch)
            q_values_next_batch =self.predict(next_state_batch)
            # 将奖励的列表转换为一个numpy数组，然后转换为一个一维的张量
            reward_batch = torch.from_numpy(np.array(reward_batch, dtype=np.float32)[:, None])
            action_batch = torch.from_numpy(np.array([[1, 0] if action == 0 else [0, 1]
                                                      for action in action_batch], dtype=np.float32))
            # 把奖励和动作也放到gpu上
            reward_batch = reward_batch.to(self.device)
            action_batch = action_batch.to(self.device)

            # 计算预测的Qvalue
            q_value = torch.sum(q_values_batch * action_batch, dim=1)

            # 计算目标的动作值，将奖励和下一个状态的最大Qvalue相加
            td_targets = torch.cat(tuple(reward if terminal else reward + gamma * torch.max(prediction)
                                         for reward, terminal, prediction
                                         in zip(reward_batch, done_batch, q_values_next_batch)))

            loss = self.update(q_value, td_targets)
            return loss


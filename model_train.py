import torch
import numpy as np
import random
from collections import deque
import FlappyBird
import cv2
import os
from DQN import DQN


image_size = 84
batch_size = 32
lr = 1e-6
gamma = 0.99
init_epsilon = 0.1
final_epsilon = 1e-4
n_iter = 2000000
memory_size = 50000
n_action = 2
screen_width = 288
saved_path = './trained_models/'

# 图像预处理
def pre_processing(image, width, height):
    # 对图像进行缩放，转换为灰度图，二值化，返回一个浮点数类型的tensor，形状为(1, height, width)
    image = cv2.cvtColor(cv2.resize(image, (width, height)), cv2.COLOR_BGR2GRAY)
    _, image = cv2.threshold(image, 1, 255, cv2.THRESH_BINARY)
    return image[None, :, :].astype(np.float32)

# epsilon贪婪策略
def gen_epsilon_greedy_policy(estimator, epsilon, n_action):
    def policy_function(state):
        if random.random() < epsilon:
            # 如果随机数小于epsilon，表示进行探索，随机返回一个动作
            return random.randint(0, n_action - 1)
        else:
            # 否则调用predict方法，根据状态预测动作的值，返回最大值对应的动作
            q_values = estimator.predict(state)
            return torch.argmax(q_values).item()
    return policy_function


if __name__ == '__main__':
    # 1.初始化DNQ
    torch.manual_seed(123)
    estimator = DQN(n_action)
    # 2.初始化游戏
    # 内存队列
    memory = deque(maxlen=memory_size)
    # 游戏环境
    env = FlappyBird.FlappyBird()
    image, reward, is_done = env.next_step(0)
    image = pre_processing(image[:screen_width, :int(env.base_y)], image_size, image_size)
    image = torch.from_numpy(image)
    # 将图像复制四次，拼接成一个tensor，作为初始的状态，形状为(1, 4, image_size, image_size)
    state = torch.cat(tuple(image for _ in range(4)))[None, :, :, :]

    for iter in range(n_iter):
        print(os.path.join(saved_path, str(iter + 1) + '.pt'))
        # 计算ε，根据迭代次数线性衰减
        epsilon = final_epsilon + (n_iter - iter)*(init_epsilon - final_epsilon) / n_iter
        # 生成一个epsilon贪婪策略，参数为estimator，epsilon和动作的数量
        policy = gen_epsilon_greedy_policy(estimator, epsilon, n_action)
        # 根据策略和状态得到一个动作
        action = policy(state)

        next_image, reward, is_done = env.next_step(action)
        next_image = pre_processing(next_image[:screen_width, :int(env.base_y)], image_size, image_size)
        next_image = torch.from_numpy(next_image)
        # 将状态的第二到第四个图像和下一个图像拼接成一个张量，作为下一个状态，
        next_state = torch.cat((state[0, 1:, :, :], next_image))[None, :, :, :]
        # 添加记忆
        memory.append([state, action, next_state, reward, is_done])

        # 从记忆中采样一批数据，进行经验回放，更新模型的权重
        loss = estimator.replay(memory, batch_size, gamma)
        state = next_state
        if (iter + 1) % 10 == 0:
            print("Iteration: {}/{}, Action: {},Loss: {}, Epsilon{}, Reward: {}"
                  .format(iter + 1, n_iter, action, loss, epsilon, reward))

        if (iter + 1) % 10000 == 0:
            torch.save(estimator.model.state_dict(), os.path.join(saved_path, str(iter + 1) + '.pt'))

    torch.save(estimator.model.state_dict(), os.path.join(saved_path, 'final.pt'))



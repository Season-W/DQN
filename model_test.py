import torch
import os
import FlappyBird
import DQN
from model_train import pre_processing


image_size = 84
screen_width = 288

if __name__ == '__main__':
    # 导入网络结构
    model = DQN.DQNModel()
    # 导入网络的参数
    model.load_state_dict(torch.load('./trained_models/final.pt'))

    n_episode = 100
    for episode in range(n_episode):
        env = FlappyBird.FlappyBird()
        image, reward, is_done = env.next_step(0)
        image = pre_processing(image[:screen_width, :int(env.base_y)], image_size, image_size)
        image = torch.from_numpy(image)
        state = torch.cat(tuple(image for _ in range(4)))[None, :, :, :]

        while True:
            prediction = model(state)[0]
            action = torch.argmax(prediction).item()
            next_image, reward, is_done = env.next_step(action)

            if is_done:
                break

            next_image = pre_processing(next_image[:screen_width,
                                :int(env.base_y)], image_size, image_size)
            next_image = torch.from_numpy(next_image)
            next_state = torch.cat((state[0, 1:, :, :], next_image))[None, :, :, :]
            state = next_state
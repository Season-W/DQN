from pygame.image import load
from pygame.surfarray import pixels_alpha
from pygame.transform import rotate
from itertools import cycle
from random import randint
import pygame

# 导入游戏图片
def load_images(sprites_path):
    base_image = load(sprites_path + 'base.png').convert_alpha()
    background_image = load(sprites_path + 'background-black.png').convert()
    pipe_images = [rotate(load(sprites_path + 'pipe-green.png').convert_alpha(), 180),
                   load(sprites_path + 'pipe-green.png').convert_alpha()]
    bird_images = [load(sprites_path + 'redbird-upflap.png').convert_alpha(),
                   load(sprites_path + 'redbird-midflap.png').convert_alpha(),
                   load(sprites_path + 'redbird-downflap.png').convert_alpha()]
    bird_hitmask = [pixels_alpha(image).astype(bool) for image in bird_images]
    pipe_hitmask = [pixels_alpha(image).astype(bool) for image in pipe_images]
    return base_image, background_image, pipe_images, bird_images, bird_hitmask, pipe_hitmask

# 初始化游戏
pygame.init()
# 初始化时钟，设置30帧/秒作为屏幕刷新频率
fps_clock = pygame.time.Clock()
fps = 30

# 屏幕大小，标题
screen_width = 288
screen_height = 512
screen = pygame.display.set_mode((screen_width, screen_height))
pygame.display.set_caption('Flappy Bird')

base_image, background_image, pipe_images, bird_images, bird_hitmask, pipe_hitmask = load_images('assets/sprites/')

# 获取游戏相关值
bird_width = bird_images[0].get_width() 
bird_height = bird_images[0].get_height() 
pipe_width = pipe_images[0].get_width() 
pipe_height = pipe_images[0].get_height ()
# 上下管道间隔
pipe_gap_size = 100
# 小鸟动作循环
bird_index_gen = cycle([0, 1, 2, 1])

class FlappyBird(object):
    def __init__(self):
        # 管道水平速度
        self.pipe_vel_x = -4
        # 小鸟最小和最大垂直速度
        self.min_velocity_y = -8
        self.max_velocity_y = 10
        # 小鸟向上和向下的加速度
        self.downward_speed = 1
        self.upward_speed = -9
        # 小鸟当前垂直速度
        self.cur_velocity_y = 0
        # 迭代器
        self.iter = 0
        # 小鸟当前动作
        self.bird_index = 0
        # 分数
        self.score = 0
        # 小鸟位置
        self.bird_x = int(screen_width / 5)
        self.bird_y = int((screen_height - bird_height) / 2)
        # 地面位置
        self.base_x = 0
        self.base_y = screen_height * 0.79
        # 地面水平偏移量
        self.base_shift = base_image.get_width() - background_image.get_width()
        # 管道
        self.pipes = [self.gen_random_pipe(screen_width), self.gen_random_pipe(screen_width * 1.5)]
        # 小鸟是否飞行
        self.is_flapped = False

    # 生成随即管道方法
    def gen_random_pipe(self, x):
        gap_y = randint(2, 10) * 10 + int(self.base_y * 0.2)
        return {"x_upper": x, "y_upper": gap_y - pipe_height,
                "x_lower": x, "y_lower": gap_y + pipe_gap_size}

    # 检查是否碰撞
    def check_collision(self):
        # 小鸟是否落地
        if bird_height + self.bird_y >= self.base_y - 1:
            return True
        # 小鸟的边界框，参数是小鸟的水平和垂直位置，宽度和高度
        bird_rect = pygame.Rect(self.bird_x, self.bird_y, bird_width, bird_height)
        # 遍历管道
        for pipe in self.pipes:
            pipe_boxes = [pygame.Rect(pipe["x_upper"], pipe["y_upper"], pipe_width, pipe_height),
                          pygame.Rect(pipe["x_lower"], pipe["y_lower"], pipe_width, pipe_height)]
            # 检查小鸟的边界框是否与任何管道的边界框重叠
            if bird_rect.collidelist(pipe_boxes) == -1:
                return False
            # 若有重叠，遍历上下两个管道的边界框
            for i in range(2):
                # 重叠区域边界框
                cropped_bbox = bird_rect.clip(pipe_boxes[i])
                x1 = cropped_bbox.x - bird_rect.x
                y1 = cropped_bbox.y - bird_rect.y
                x2 = cropped_bbox.x - pipe_boxes[i].x
                y2 = cropped_bbox.y - pipe_boxes[i].y

                # 遍历重叠区域的每个像素点
                for x in range(cropped_bbox.width):
                    for y in range(cropped_bbox.height):
                        if bird_hitmask[self.bird_index][x1 + x, y1 + y] and pipe_hitmask[i][x2 + x, y2 + y]:
                            return True
        
        return False

    # 根据action更新界面
    def next_step(self, action):
        pygame.event.pump()
        # 小鸟存活奖励
        reward = 0.1
        # 小鸟飞行时
        if action == 1:
            # 垂直速度为向上加速度
            self.cur_velocity_y = self.upward_speed
            self.is_flapped = True

        # 小鸟水平中心位置
        bird_center_x = self.bird_x + bird_width / 2
        for pipe in self.pipes:
            pipe_center_x = pipe["x_upper"] + pipe_width / 2
            # 小鸟通过管道，+1分
            if pipe_center_x < bird_center_x < pipe_center_x + 5:
                self.score += 1
                reward = 1
                break

        # 小鸟是否切换动画
        if (self.iter + 1) % 3 == 0:
            self.bird_index = next(bird_index_gen)
        self.iter = (self.iter + 1) % fps
        # 地面的水平位置等于地面的水平位置加100对地面的移动距取余
        self.base_x = -((-self.base_x + 100) % self.base_shift)

        # 更新小鸟向下加速度
        if self.cur_velocity_y < self.max_velocity_y and not self.is_flapped:
            self.cur_velocity_y += self.downward_speed
        self.is_flapped = False
        # 更新小鸟垂直位置
        self.bird_y += min(self.cur_velocity_y, self.bird_y - self.cur_velocity_y - bird_height)
        if self.bird_y < 0:
            self.bird_y = 0

        # 更新管道位置
        for pipe in self.pipes:
            pipe["x_upper"] += self.pipe_vel_x
            pipe["x_lower"] += self.pipe_vel_x

        # 当第一个管道即将触及屏幕左侧时添加新管道
        if 0 < self.pipes[0]["x_lower"] < 5:
            self.pipes.append(self.gen_random_pipe(screen_width + 10))

        # 如果第一个管道超出屏幕，则将其移除
        if self.pipes[0]["x_lower"] < -pipe_width:
            self.pipes.pop(0)

        if self.check_collision():
            is_done = True
            reward = -1
            self.__init__()
        else:
            is_done = False

        # 绘制精灵
        screen.blit(background_image, (0, 0))
        screen.blit(base_image, (self.base_x, self.base_y))
        screen.blit(bird_images[self.bird_index], (self.bird_x, self.bird_y))
        for pipe in self.pipes:
            screen.blit(pipe_images[0], (pipe["x_upper"], pipe["y_upper"]))
            screen.blit(pipe_images[1], (pipe["x_lower"], pipe["y_lower"]))

        image = pygame.surfarray.array3d(pygame.display.get_surface())
        pygame.display.update()
        fps_clock.tick(fps)
        return image, reward, is_done

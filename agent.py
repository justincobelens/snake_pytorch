import numpy
import torch
import random
import numpy as np
from collections import deque

from game import SnakeGame, Direction, Point, BLOCK_SIZE
from model import LinearQnet, QTrainer
from helper import create_plots, update_plots

MAX_MEMORY = 100_000
BATCH_SIZE = 1_000
LR = 0.001
EPOCHS = 2_000


# TODO: Give the snake more states, some ideas:
#  1. [DONE] Distance to food, something like its Euclidean distance between head and food
#  1. [DONE] maybe using bins to group distances together so it can be represented as bools
#  2. [DONE] Free space, calculate free blocks around snakes head
#  3. Body direction, representation of second block to the head
#  4. Tail position or tail relative position, so it stops endlessly chasing its tail
#  5. Foods position relative to body instead of head
#  6. Curvature of snake, something like the angle between head and first to following body part


class Agent:

    def __init__(self):
        self.n_games = 0  # number of games
        self.epsilon = 0  # randomness
        self.gamma = 0.9  # discount rate
        self.memory = deque(maxlen=MAX_MEMORY)  # popleft() if MAX_MEMORY is reached

        self.model = LinearQnet(20, 256, 256, 128, 3)  # (states, hidden_size, action(forward, left, right))
        # TODO: Load model if one exists or load checkpoint
        # self.model.load()
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)

    @staticmethod
    def get_state(game):
        head = game.snake[0]
        snake_body = game.snake[1:]

        # direction of snake
        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        # next points
        point_l = Point(head.x - BLOCK_SIZE, head.y)  # left
        point_r = Point(head.x + BLOCK_SIZE, head.y)  # right
        point_u = Point(head.x, head.y - BLOCK_SIZE)  # up
        point_d = Point(head.x, head.y + BLOCK_SIZE)  # down

        # Free space around snake's head
        points = [point_l, point_r, point_u, point_d]
        snake_positions = set((segment.x, segment.y) for segment in game.snake)
        free_cells_around_head = [1 if not game.is_collision(point) and (point.x, point.y) not in snake_positions else 0
                                  for point in points]

        # point awareness, free points around snakes head
        points = [point_l, point_r, point_u, point_d]
        free_points_around_head = sum(1 for point in points if not game.is_collision(point))
        normalized_free_cells = free_points_around_head // 4

        # euclidean distance to food in bins
        bins = 4  # CAN CHANGE, maximum bins(groups of blocks) to be considered
        max_distance = 100  # CAN CHANGE, maximum distance to be considered
        bin_width = max_distance / bins
        distance = np.sqrt((game.head.x - game.food.x) ** 2 + (game.head.y - game.food.y) ** 2)
        bin_idx = min(int(distance // bin_width), bins - 1)  # prevents idx to be bigger than bins
        one_hot_distance = [0] * bins
        one_hot_distance[bin_idx] = 1

        # euclidean distance to tail in bins
        # bins = 4  # CAN CHANGE, maximum bins(groups of blocks) to be considered
        # max_distance = 100  # CAN CHANGE, maximum distance to be considered
        # bin_width = max_distance / bins
        # distance = np.sqrt((game.head.x - game.food.x) ** 2 + (game.head.y - game.food.y) ** 2)
        # bin_idx = min(int(distance // bin_width), bins - 1)  # prevents idx to be bigger than bins
        # one_hot_distance = [0] * bins
        # one_hot_distance[bin_idx] = 1

        state = [
            # Danger straight (checks if direction moves into a collision)
            (dir_r and game.is_collision(point_r)) or
            (dir_l and game.is_collision(point_l)) or
            (dir_u and game.is_collision(point_u)) or
            (dir_d and game.is_collision(point_d)),

            # Danger right
            (dir_u and game.is_collision(point_r)) or
            (dir_d and game.is_collision(point_l)) or
            (dir_l and game.is_collision(point_u)) or
            (dir_r and game.is_collision(point_d)),

            # Danger left
            (dir_d and game.is_collision(point_r)) or
            (dir_u and game.is_collision(point_l)) or
            (dir_r and game.is_collision(point_u)) or
            (dir_l and game.is_collision(point_d)),

            # Move direction, only 1 is True
            dir_l,
            dir_r,
            dir_u,
            dir_d,

            # Food location rel to head
            game.food.x < game.head.x,  # food left
            game.food.x > game.head.x,  # food right
            game.food.y < game.head.y,  # food up
            game.food.y > game.head.y,  # food down

            # Distance to food (binned and one-hot encoded)
            *one_hot_distance,

            # Point awareness
            normalized_free_cells,

            # Free space around head
            *free_cells_around_head
        ]
        return np.array(state, dtype=int)  # change bools to 1 or 0

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))  # stores 1 tuple in memory deque

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)  # return list of tuples
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)  # unpack and zip
        return self.train_short_memory(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        return self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        # random moves: tradeoff between exploration / exploitation

        self.epsilon = 200 - self.n_games  # change if needed
        final_move = [0, 0, 0]

        # the tradeoff logic
        if random.randint(0, 500) < self.epsilon:  # random move
            move = random.randint(0, 2)
            final_move[move] = 1
        else:  # predicted move
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        return final_move


def train():
    plot_scores = []
    plot_mean_scores = []

    total_score = 0
    record = 0

    train_loss_values = []

    agent = Agent()
    game = SnakeGame()

    fig, axs = create_plots()

    while True:
        #  get old state
        state_old = agent.get_state(game)

        # get move
        final_move = agent.get_action(state_old)

        # perform move and get new state
        reward, done, score = game.play_step(final_move, record, agent.n_games)
        state_new = agent.get_state(game)

        # train short memory
        agent.train_short_memory(state_old, final_move, reward, state_new, done)

        # remember
        agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            # train long memory, plot results
            game.reset()
            agent.n_games += 1
            loss = agent.train_long_memory()

            # TODO: save checkpoints instead of model
            if score > record:
                record = score
                agent.model.save(file_name='model3.pth')
                print(f"\nSAVING.. Saving on epoch: {agent.n_games} with record: {record}\n")

            # TODO: save checkpoints instead of model
            if EPOCHS:
                if agent.n_games > EPOCHS:
                    print(f'STOPPING.. Reached EPOCH: {agent.n_games}/{EPOCHS} with record: {record}.')
                    break

            print(f'Game: {agent.n_games} | Score: {score} | Record: {record}')

            train_loss_values.append(loss)
            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            update_plots(fig, axs, plot_scores, plot_mean_scores, train_loss_values)


if __name__ == '__main__':
    train()

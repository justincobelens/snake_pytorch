import torch
import random
import numpy as np
from collections import deque
from game import SnakeGame, Direction, Point, BLOCK_SIZE
from model import Linear_Qnet, QTrainer
from helper import plot


MAX_MEMORY = 100_000
BATCH_SIZE = 1_000
LR = 0.001


class Agent:

    def __init__(self):
        self.n_games = 0  # number of games
        self.epsilon = 0  # randomness
        self.gamma = 0.9  # discount rate
        self.memory = deque(maxlen=MAX_MEMORY)  # popleft() if MAX_MEMORY is reached

        self.model = Linear_Qnet(11, 256, 3)  # (states, hidden_size, action(forward, left, right))
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)

        #  TODO: model, trainer

    def get_state(self, game):
        head = game.snake[0]

        point_l = Point(head.x - BLOCK_SIZE, head.y)
        point_r = Point(head.x + BLOCK_SIZE, head.y)
        point_u = Point(head.x, head.y - BLOCK_SIZE)
        point_d = Point(head.x, head.y + BLOCK_SIZE)

        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        state = [
            # Danger straight (checks if direction moves into a collision)
            (dir_r and game.is_collision(point_r)) or
            (dir_l and game.is_collision(point_l)) or
            (dir_u and game.is_collision(point_u)) or
            (dir_d and game.is_collision(point_d)),

            # Danger right
            (dir_r and game.is_collision(point_r)) or
            (dir_l and game.is_collision(point_l)) or
            (dir_u and game.is_collision(point_u)) or
            (dir_d and game.is_collision(point_d)),

            # Danger left
            (dir_u and game.is_collision(point_r)) or
            (dir_d and game.is_collision(point_l)) or
            (dir_l and game.is_collision(point_u)) or
            (dir_r and game.is_collision(point_d)),

            # Danger down
            (dir_d and game.is_collision(point_r)) or
            (dir_u and game.is_collision(point_l)) or
            (dir_r and game.is_collision(point_u)) or
            (dir_l and game.is_collision(point_d)),

            # Move direction, only 1 is True
            dir_l,
            dir_r,
            dir_u,
            dir_d,

            # Food location, gives direction of food
            game.food.x < game.head.x,  # food left
            game.food.x < game.head.x,  # food right
            game.food.y < game.head.y,  # food up
            game.food.y < game.head.y,  # food down
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
        self.train_short_memory(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        # random moves: tradeoff between exploration / exploitation

        self.epsilon = 80 - self.n_games  # change if needed
        final_move = [0, 0, 0]

        # the tradeoff logic
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
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

    agent = Agent()
    game = SnakeGame()

    while True:
        #  get old state
        state_old = agent.get_state(game)

        # get move
        final_move = agent.get_action(state_old)

        # perform move and get new state
        reward, done, score = game.play_step(final_move)
        state_new = agent.get_state(game)

        # train short memory
        agent.train_short_memory(state=state_old, action=final_move, reward=reward, next_state=state_new, done=done)

        # remember
        agent.remember(state=state_old, action=final_move, reward=reward, next_state=state_new, done=done)

        if done:
            # train long memory, plot results
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()

            if score > record:
                record = score
                agent.model.save()

            print(f'Game: {agent.n_games} | Score: {score} | Record: {record}')

            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores)


if __name__ == '__main__':
    train()

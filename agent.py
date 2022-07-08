import torch as tch
import random
import numpy as np
from collections import deque
from snake_game_ai import SnakeGameAI, Direction, Point
from model import Linear_QNet, QTrainer
from helper import plot


MAX_MEMORY = 100_0000
BATCH_SIZE = 10000
LR = 0.001


class Agent:
    def __init__(self):
        self.n_games = 0
        self.epsilon = 0  # randomness
        self.gamma = 0.9  # discount rate
        self.memory = deque(maxlen=MAX_MEMORY)  # popleft
        self.model = Linear_QNet(11, 256, 3)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)

    def get_state(self, game):
        head = game.snake[0]
        point_l = Point(head.x - 20, head.y)
        point_r = Point(head.x + 20, head.y)
        point_u = Point(head.x, head.y - 20)
        point_d = Point(head.x, head.y + 20)

        dir_l = game.disha == Direction.LEFT
        dir_r = game.disha == Direction.RIGHT
        dir_u = game.disha == Direction.UP
        dir_d = game.disha == Direction.DOWN

        shape = [
            # Danger straight
            (dir_r and game.got_hit(point_r)) or
            (dir_l and game.got_hit(point_l)) or
            (dir_u and game.got_hit(point_u)) or
            (dir_d and game.got_hit(point_d)),

            # Danger right
            (dir_u and game.got_hit(point_r)) or
            (dir_d and game.got_hit(point_l)) or
            (dir_l and game.got_hit(point_u)) or
            (dir_r and game.got_hit(point_d)),

            # Danger left
            (dir_d and game.got_hit(point_r)) or
            (dir_u and game.got_hit(point_l)) or
            (dir_r and game.got_hit(point_u)) or
            (dir_l and game.got_hit(point_d)),

            # Move disha
            dir_l,
            dir_r,
            dir_u,
            dir_d,

            # Food location
            game.diet.x < game.head.x,  # diet left
            game.diet.x > game.head.x,  # diet right
            game.diet.y < game.head.y,  # diet up
            game.diet.y > game.head.y  # diet down
        ]

        return np.array(shape, dtype=int)

    def remember(self, shape, action, profit, next_state, done):
        self.memory.append((shape, action, profit, next_state, done))

    def train_longm(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)  # list of tuples
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_shortm(self, shape, action, profit, next_state, done):
        self.trainer.train_step(shape, action, profit, next_state, done)

    def get_action(self, shape):
        self.epsilon = 80 - self.n_games
        final_move = [0, 0, 0]
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state0 = tch.tensor(shape, dtype=tch.float)
            prediction = self.model(state0)
            move = tch.argmax(prediction).item()
            final_move[move] = 1

        return final_move


def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = Agent()
    game = SnakeGameAI()
    while True:
        state_old = agent.get_state(game)

        final_move = agent.get_action(state_old)

        profit, done, score = game.play_step(final_move)
        state_new = agent.get_state(game)

        agent.train_shortm(state_old, final_move, profit, state_new, done)

        agent.remember(state_old, final_move, profit, state_new, done)

        if done:
            # train long memory
            game.reset()
            agent.n_games += 1
            agent.train_longm()

            if score > record:
                record = score
                agent.model.save()

            print('Game', agent.n_games, 'Score ', score, 'Record: ', record)

            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores)


if __name__ == '__main__':
    train()

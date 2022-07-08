import pygame
import random
from enum import Enum
from collections import namedtuple
import numpy as np

pygame.init()
font = pygame.font.Font('fontitalic.otf', 25)

#reset

#profit
#play(action) ->disha
#game iteration
# is collision

class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4

Point = namedtuple('Point', 'x , y')

WHITE = (255, 255, 255)
RED = (200, 0, 0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)
BLACK = (0, 0, 0)

BLOCK_SIZE = 20
SPEED = 200

class SnakeGameAI:
    def __init__(saanp,w=640,h=480):
        saanp.w=w
        saanp.h=h

        saanp.display = pygame.display.set_mode((saanp.w,saanp.h))
        pygame.display.set_caption('Snake')
        saanp.clock = pygame.time.Clock()
        saanp.reset()

    def reset(saanp):
        # initial game position
        saanp.disha = Direction.RIGHT
        saanp.head = Point(saanp.w / 2, saanp.h / 2)
        saanp.snake = [saanp.head, Point(saanp.head.x - BLOCK_SIZE, saanp.head.y),
                      Point(saanp.head.x - (2 * BLOCK_SIZE), saanp.head.y)]
        saanp.score = 0
        saanp.diet = None
        saanp._place_food()
        saanp.frame_iteration = 0

    def _place_food(saanp):
        x = random.randint(0, (saanp.w - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        y = random.randint(0, (saanp.h - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        saanp.diet = Point(x, y)
        if saanp.diet in saanp.snake:
            saanp._place_food()


    def play_step(saanp,action):
        saanp.frame_iteration +=1
        # 1.yaha user input lele
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    saanp.disha = Direction.LEFT
                elif event.key == pygame.K_RIGHT:
                    saanp.disha = Direction.RIGHT
                elif event.key == pygame.K_UP:
                    saanp.disha = Direction.UP
                elif event.key == pygame.K_DOWN:
                    saanp.disha = Direction.DOWN
        # 2. saanp ko chala
        saanp._move(action)
        saanp.snake.insert(0,saanp.head)
        # 3. game over check
        profit = 0
        game_over = False
        if saanp.got_hit() or saanp.frame_iteration > 100 * len(saanp.snake):
            game_over=True
            profit = -10
            return profit , game_over , saanp.score
        # 4. new diet rakhde
        if saanp.head == saanp.diet:
            saanp.score += 1
            profit = 10
            saanp._place_food()
        else:
            saanp.snake.pop()
        # 5. update ui and clock
        saanp._update_ui()
        saanp.clock.tick(SPEED)
        # 6. return game over and clock
        return profit , game_over , saanp.score

    def got_hit(saanp, pt=None):
        if pt is None:
            pt = saanp.head
        # hits boundary
        if pt.x > saanp.w - BLOCK_SIZE or pt.x < 0 or pt.y > saanp.h -BLOCK_SIZE or pt.y < 0:
            return True
        # hits itsaanp
        if pt in saanp.snake[1:]:
            return True

        return False

    def _update_ui(saanp):
        saanp.display.fill(BLACK)

        for pt in saanp.snake:
            pygame.draw.rect(saanp.display,BLUE1,pygame.Rect(pt.x,pt.y,BLOCK_SIZE,BLOCK_SIZE))
            pygame.draw.rect(saanp.display, BLUE2, pygame.Rect(pt.x+4, pt.y+4, 12 , 12))

        pygame.draw.rect(saanp.display,RED ,  pygame.Rect(saanp.diet.x, saanp.diet.y, BLOCK_SIZE, BLOCK_SIZE))

        text = font.render("Score: " + str(saanp.score), True, WHITE)
        saanp.display.blit(text,[0,0])
        pygame.display.flip()

    def _move(saanp,action):
        #[straight,right,left]
        clock_wise = [Direction.RIGHT , Direction.DOWN , Direction.LEFT , Direction.UP]
        idx = clock_wise.index(saanp.disha)

        if np.array_equal(action , [1,0,0]):
            new_dir = clock_wise[idx]
        elif np.array_equal(action , [0,1,0]):
            next_idx = (idx+1) % 4
            new_dir = clock_wise[next_idx]
        else:
            next_idx = (idx - 1) % 4
            new_dir = clock_wise[next_idx]

        saanp.disha = new_dir

        x = saanp.head.x
        y = saanp.head.y
        if saanp.disha == Direction.RIGHT:
            x += BLOCK_SIZE
            # x = x % saanp.w
        elif saanp.disha == Direction.LEFT:
            x -= BLOCK_SIZE
            # if(x<0):
            #     x+=saanp.w
        elif saanp.disha == Direction.DOWN:
            y += BLOCK_SIZE
            # y = y % saanp.h
        elif saanp.disha == Direction.UP:
            y -= BLOCK_SIZE
            # if (y < 0):
            #     y += saanp.h

        saanp.head = Point(x,y)

#
# if __name__ == '__main__':
#     game =SnakeGame()
#
#     while True:
#         game_over,score = game.play_step()
#         if game_over == True:
#             break
#     print('Your final score',score)
#
#     pygame.quit()

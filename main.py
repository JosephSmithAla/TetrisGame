import keyboard

from CNNModel import TetrisModel
from Tetris import TetrisGameInstance
import NeuroevolutionModel
import random
import pygame
import numpy


BLOCK_SIZE = 30
GRID_WIDTH, GRID_HEIGHT = 20, 20
HEADER_HEIGHT = 60

SCREEN_WIDTH = GRID_WIDTH * BLOCK_SIZE
SCREEN_HEIGHT = (GRID_HEIGHT * BLOCK_SIZE) + HEADER_HEIGHT

pygame.init()
pygame.font.init()
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("AI Tetris - NE vs CNN")
clock = pygame.time.Clock()
title_font = pygame.font.SysFont("Arial", 18, bold=True)
score_font = pygame.font.SysFont("Arial", 16)


def visualize(NEGame, CNNGame, fps=5):
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            return

    screen.fill((15, 15, 20))

    ne_score = NEGame.LinesCleared
    cnn_score = CNNGame.LinesCleared

    half_width = SCREEN_WIDTH // 2

    ne_title = title_font.render("Neuroevolution", True, (0, 200, 255))
    ne_score_text = score_font.render(f"Score: {ne_score}", True, (220, 220, 220))
    screen.blit(ne_title, (half_width // 2 - ne_title.get_width() // 2, 10))
    screen.blit(ne_score_text, (half_width // 2 - ne_score_text.get_width() // 2, 34))

    cnn_title = title_font.render("CNN", True, (255, 140, 0))
    cnn_score_text = score_font.render(f"Score: {cnn_score}", True, (220, 220, 220))
    screen.blit(cnn_title, (half_width + half_width // 2 - cnn_title.get_width() // 2, 10))
    screen.blit(cnn_score_text, (half_width + half_width // 2 - cnn_score_text.get_width() // 2, 34))


    pygame.draw.line(screen, (50, 50, 60), (half_width, 0), (half_width, SCREEN_HEIGHT), 2)

    pygame.draw.line(screen, (50, 50, 60), (0, HEADER_HEIGHT), (SCREEN_WIDTH, HEADER_HEIGHT), 2)

    board = numpy.concatenate((NEGame.PlayingGround, CNNGame.PlayingGround), axis=1)

    if board is not None:
        for r in range(GRID_HEIGHT):
            for c in range(GRID_WIDTH):
                if board[r][c] != 0:
                    rect = pygame.Rect(
                        c * BLOCK_SIZE,
                        HEADER_HEIGHT + (r * BLOCK_SIZE),
                        BLOCK_SIZE - 1,
                        BLOCK_SIZE - 1
                    )
                    color = (0, 200, 255) if c < 10 else (255, 140, 0)
                    pygame.draw.rect(screen, color, rect)

    pygame.display.flip()
    clock.tick(fps)



CNNGame = TetrisGameInstance()
NEGame = TetrisGameInstance()

NEGame.StartGame()
CNNGame.StartGame()

NEModel = NeuroevolutionModel.ModelManager(100)
CNNModel = TetrisModel(1e-4, 100, 0.9995)
CNNModel.load()

cnndead = False
nedead = False

while not (cnndead and nedead):
    seed = random.randint(1, 100)
    CNNGame.SetSeed(seed)
    NEGame.SetSeed(seed)
    if not nedead:
        ne_score = NEGame.LinesCleared
        nedata = NEGame.GameLoopCNN(*NEModel.DecideMove(NEGame.PlayingGround, NEGame.MyTetromino.Type, NEGame))
        nedead = nedata[3]
        if nedead:
            NEGame.LinesCleared, NEGame.PlayingGround = ne_score, nedata[0]
        else:
            NEGame.LinesCleared += nedata[1]
    if not cnndead:
        cnn_score = CNNGame.LinesCleared
        cnndata = CNNGame.GameLoopCNN(*CNNModel.decide(CNNGame.PlayingGround, CNNGame.MyTetromino.Type))
        cnndead = cnndata[3]
        if cnndead:
            CNNGame.LinesCleared, CNNGame.PlayingGround = cnn_score, cnndata[0]
        else:
            CNNGame.LinesCleared += cnndata[1]
    #GUI
    visualize(NEGame, CNNGame, fps=10)
pygame.quit()

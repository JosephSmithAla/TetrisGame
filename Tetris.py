import time

import numpy as np
import random
import pygame


class Tetromino:
    def __init__(self, tetromino_type="T", main_position=np.array([0, 0])):
        self.Type = tetromino_type
        self.Position = main_position
        self.Rotation : int
        self.RotationMaxCount : int
        self.TETROMINO_SHAPES = {
            "I": [
                [(0, 0), (0, 1), (0, -1), (0, -2)],
                [(0, 0), (-1, 0), (1, 0), (2, 0)],
            ],
            "L": [
                [(0, 0), (0, 1), (0, -1), (1, -1)],
                [(0, 0), (-1, 0), (1, 0), (1, 1)],
                [(0, 0), (0, -1), (0, 1), (-1, 1)],
                [(0, 0), (1, 0), (-1, 0), (-1, -1)],
            ],
            "S": [
                [(0, 0), (1, 0), (0, -1), (-1, -1)],
                [(0, 0), (0, 1), (1, 0), (1, -1)],
            ],
            "Z": [
                [(0, 0), (-1, 0), (0, -1), (1, -1)],
                [(0, 0), (0, 1), (-1, 0), (-1, -1)],
            ],
            "O": [[(0, 0), (1, 0), (0, -1), (1, -1)]],
            "J": [
                [(0, 0), (0, 1), (0, -1), (-1, -1)],
                [(0, 0), (1, 0), (-1, 0), (1, -1)],
                [(0, 0), (0, -1), (0, 1), (1, 1)],
                [(0, 0), (-1, 0), (1, 0), (-1, 1)],
            ],
            "T": [
                [(0, 0), (1, 0), (-1, 0), (0, -1)],
                [(0, 0), (0, 1), (0, -1), (1, 0)],
                [(0, 0), (-1, 0), (1, 0), (0, 1)],
                [(0, 0), (0, -1), (0, 1), (-1, 0)],
            ],
        }
        self.main_piece : int
        self.sub_piece1 : int
        self.sub_piece2 : int
        self.sub_piece3 : int
        self.Construct()

    def Construct(self):
        match self.Type:
            case "I":
                self.main_piece = np.array([0, 0])
                self.sub_piece1 = np.array([0, 1])
                self.sub_piece2 = np.array([0, -1])
                self.sub_piece3 = np.array([0, -2])
            case "L":
                self.main_piece = np.array([0, 0])
                self.sub_piece1 = np.array([0, 1])
                self.sub_piece2 = np.array([0, -1])
                self.sub_piece3 = np.array([1, -1])
            case "S":
                self.main_piece = np.array([0, 0])
                self.sub_piece1 = np.array([1, 0])
                self.sub_piece2 = np.array([0, -1])
                self.sub_piece3 = np.array([-1, -1])
            case "Z":
                self.main_piece = np.array([0, 0])
                self.sub_piece1 = np.array([-1, 0])
                self.sub_piece2 = np.array([0, -1])
                self.sub_piece3 = np.array([1, -1])
            case "O":
                self.main_piece = np.array([0, 0])
                self.sub_piece1 = np.array([1, 0])
                self.sub_piece2 = np.array([0, -1])
                self.sub_piece3 = np.array([1, -1])
            case "J":
                self.main_piece = np.array([0, 0])
                self.sub_piece1 = np.array([0, 1])
                self.sub_piece2 = np.array([0, -1])
                self.sub_piece3 = np.array([-1, -1])
            case "T":
                self.main_piece = np.array([0, 0])
                self.sub_piece1 = np.array([1, 0])
                self.sub_piece2 = np.array([-1, 0])
                self.sub_piece3 = np.array([0, -1])

        self.RotationMaxCount = len(self.TETROMINO_SHAPES[self.Type])



    def GetPiecesLocation(self, main_position):
        return [main_position + self.main_piece, main_position + self.sub_piece1, main_position + self.sub_piece2,
                main_position + self.sub_piece3]


    def GetRotatedPositions(self, rotate_count):
        locs = np.array(self.TETROMINO_SHAPES[self.Type][(self.Rotation + rotate_count) % self.RotationMaxCount])
        return [self.Position + locs[0], self.Position + locs[1], self.Position + locs[2], self.Position + locs[3]]

    def Rotate(self, rotate_count):

        self.main_piece, self.sub_piece1, self.sub_piece2, self.sub_piece3 = self.GetRotatedPositions(rotate_count)

        self.Rotation += rotate_count



class TetrisGameInstance:
    def __init__(self):
        self.Gravity = 1
        self.BufferedInput = np.array([])
        self.DefaultPosition = np.array([5, 19])
        self.PlayingGround = np.zeros((20, 10), dtype=int)
        self.PlayerPlayingGround = np.zeros((20, 10), dtype=int)
        self.TetrominoTypes = ["I", "L", "S", "Z", "O", "J", "T"]
        self.GameSpawnSeed = self.TetrominoTypes
        self.MyTetromino = None
        self.TetrominoCounter = 0
        self.GameScreen = None
        self.TetrominoColor = None
        self.EmptySpaceColor = None
        self.LinesCleared = 0


    def getStates(self, state, piece): #canli oynanan kaydi etkilemeyen generate state fonksiyonu,zaten cnn kullanacak sadece o yuzden canliya mudahale etmeisnde sorun yok gibi
        states_to_return = []

        livePlayingGround = self.PlayingGround.copy()
        livePlayerPlayingGround = self.PlayerPlayingGround.copy()
        self.PlayerPlayingGround = np.zeros((20, 10), dtype=int)
        self.PlayingGround = state
        tetromino = Tetromino(tetromino_type=piece, main_position=np.array([5, 19]),
                              rotation=0)
        for r in range(self.TetrominoRotationBases[piece]):
            for i in range(1, 11):
                last_loc = np.array([])
                for j in range(19, -1, -1):

                    if(self.bCheckCollisionAtPosition(tetromino.GetPiecesLocation(tetromino.Position))):
                        break
                    last_loc = tetromino.GetPiecesLocation(tetromino.Position)
                if len(last_loc) > 0:
                    cpy = self.PlayingGround
                    self.DrawToPlayerPlayingGround(last_loc)
                    self.PlayingGround = self.PlayingGround + self.PlayerPlayingGround
                    cleared_lines = self.CheckLineClears(np.unique(np.array(last_loc)[:, 1]))
                    states_to_return.append((self.PlayingGround, cleared_lines))
                    self.PlayingGround = cpy
                    self.EraseFromPlayerPlayingGround(last_loc)

        self.PlayingGround = livePlayingGround.copy()
        self.PlayerPlayingGround = livePlayerPlayingGround.copy()
        return states_to_return



    def ConstructGUI(self):
        self.GameScreen = pygame.display.set_mode((400, 800))
        self.TetrominoColor = (0, 0, 0)
        self.EmptySpaceColor = (255, 255, 255)

    def DrawToPlayingGround(self, positions_to_draw):

        positions = np.atleast_2d(np.asarray(positions_to_draw))
        if positions.size == 0:
            return

        rows = 20 - positions[:, 1]
        columns = positions[:, 0] - 1

        self.PlayingGround[rows, columns] = 1

    def EraseFromPlayingGround(self, positions_to_erase):

        positions = np.atleast_2d(np.asarray(positions_to_erase))
        if positions.size == 0:
            return

        rows = 20 - positions[:, 1]
        columns = positions[:, 0] - 1

        self.PlayingGround[rows, columns] = 0

    def DrawToPlayerPlayingGround(self, positions_to_draw):

        positions = np.atleast_2d(np.asarray(positions_to_draw))
        if positions.size == 0:
            return

        rows = 20 - positions[:, 1]
        columns = positions[:, 0] - 1

        self.PlayerPlayingGround[rows, columns] = 1

    def EraseFromPlayerPlayingGround(self, positions_to_erase):

        positions = np.atleast_2d(np.asarray(positions_to_erase))
        if positions.size == 0:
            return

        rows = 20 - positions[:, 1]
        columns = positions[:, 0] - 1

        self.PlayerPlayingGround[rows, columns] = 0

    def bCheckCollisionAtPosition(self, locations):

        for location in locations:
            if location[0] < 1 or location[0] > 10 or location[1] < 1 or location[1] > 20:
                return True
            elif self.PlayingGround[(20 - location[1], location[0] - 1)] == 1:
                return True
        return False

    def UpdateControlledTetrominoPosition(self, new_position):
        if self.bCheckCollisionAtPosition(self.MyTetromino.GetPiecesLocation(new_position)):
            return False
        else:
            self.EraseFromPlayerPlayingGround(self.MyTetromino.GetPiecesLocation(self.MyTetromino.Position))
            self.MyTetromino.Position = new_position
            self.DrawToPlayerPlayingGround(self.MyTetromino.GetPiecesLocation(self.MyTetromino.Position))
            return True

    def bTrySpawnTetromino(self):

        self.MyTetromino = Tetromino(tetromino_type= self.GameSpawnSeed[self.TetrominoCounter % 7], main_position=np.array([5, 19]), rotation=0)
        self.TetrominoCounter +=1

        if self.bCheckCollisionAtPosition(self.MyTetromino.GetPiecesLocation(self.MyTetromino.Position)):
            return False
        else:
            self.DrawToPlayerPlayingGround(self.MyTetromino.GetPiecesLocation(self.MyTetromino.Position))
            return True

    def MoveRightInput(self):
        self.BufferedInput = np.array([self.MyTetromino.Position[0] + 1, self.MyTetromino.Position[1]])
        self.GameLoop()

    def MoveLeftInput(self):

        self.BufferedInput = np.array([self.MyTetromino.Position[0] - 1, self.MyTetromino.Position[1]])
        self.GameLoop()

    def MoveDownInput(self):

        self.BufferedInput = np.array([self.MyTetromino.Position[0], self.MyTetromino.Position[1] - 1])
        self.GameLoop()

    def RotateInput(self):

        self.EraseFromPlayerPlayingGround(self.MyTetromino.GetPiecesLocation(self.MyTetromino.Position))

        locs = self.MyTetromino.GetRotatedPositions(1)

        if not self.bCheckCollisionAtPosition(locs):
            self.MyTetromino.Rotate(1)
            self.DrawToPlayerPlayingGround(self.MyTetromino.GetPiecesLocation(self.MyTetromino.Position))
            self.GameLoop()


    def BringDownLines(self, clear_row):
        for i in range(clear_row, 20):
            self.PlayingGround[20 - i] = self.PlayingGround[20 - (i + 1)] # np.all dan filtre yapilabilir, numpy c ile islme yaptigi icin daha hizli checkler ve siler ama ne kadar gerekli bilmiyorum
        self.EraseFromPlayingGround([[column + 1, 20] for column in range(10)])

    def CheckLineClears(self, rows):
        ClearedLineCount = 0
        for row in rows:
            if self.PlayingGround[20 - (row - ClearedLineCount)].sum() == 10:
                self.EraseFromPlayingGround([[column + 1, row - ClearedLineCount] for column in range(10)])
                self.BringDownLines(row - ClearedLineCount)
                ClearedLineCount += 1
                self.LinesCleared += 1
        return ClearedLineCount

    def LockTetromino(self):
        piece_locations = self.MyTetromino.GetPiecesLocation(self.MyTetromino.Position)
        self.DrawToPlayingGround(piece_locations)
        self.EraseFromPlayerPlayingGround(piece_locations)

        piece_locations_y = np.unique(np.array(piece_locations)[:, 1])

        self.CheckLineClears(piece_locations_y)

    def DrawGUI(self):
        for row in range(20):
            for column in range(10):

                if self.PlayingGround[(row, column)] != 0:
                    block_color = self.TetrominoColor
                elif self.PlayerPlayingGround[(row, column)] != 0:
                    block_color = self.TetrominoColor
                else:
                    block_color = self.EmptySpaceColor

                x = column * 40
                y = row * 40

                pygame.draw.rect(self.GameScreen, block_color, pygame.Rect(x, y, 40, 40))
        pygame.display.flip()

    def GetGameCanvasArray(self):
        canvas = self.PlayingGround.copy()
        for place in self.MyTetromino.GetPiecesLocation(self.MyTetromino.Position):
            canvas[(20 - place[1], place[0] - 1)] = -1
        return canvas.flatten()

    def GameLoopCNN(self, state, lines, gui = False):
        self.PlayingGround = state
        self.DrawToPlayerPlayingGround(self.MyTetromino.GetPiecesLocation(self.MyTetromino.Position))
        if gui:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
            self.DrawGUI()
            time.sleep(0.1)
        self.EraseFromPlayerPlayingGround(self.MyTetromino.GetPiecesLocation(self.MyTetromino.Position))
        if self.bTrySpawnTetromino():
            return (state, lines, self.MyTetromino.Type, False)
        else:
            self.StartGame()
            return (state, -100, None, True) # hiper parametre olmali


    def GameLoop(self):

        #for event in pygame.event.get():
            #if event.type == pygame.QUIT:
                #pygame.quit()

        #self.DrawGUI()

        if self.BufferedInput.size > 0:
            self.UpdateControlledTetrominoPosition(self.BufferedInput)
            self.BufferedInput = np.array([])
        if self.UpdateControlledTetrominoPosition(np.array([self.MyTetromino.Position[0], self.MyTetromino.Position[1] - self.Gravity])):
            pass
        else:
            self.LockTetromino()
            if self.bTrySpawnTetromino():
                pass
            else:
                return -1 , self.LinesCleared
        return 0, self.LinesCleared


    def StartGame(self):

        self.PlayingGround = np.zeros((20, 10), dtype=int)
        self.PlayerPlayingGround = np.zeros((20, 10), dtype=int)
        random.shuffle(self.GameSpawnSeed)
        self.MyTetromino = Tetromino(tetromino_type=self.GameSpawnSeed[0], main_position=self.DefaultPosition, rotation=0)
        self.TetrominoCounter = 1
        self.LinesCleared = 0
        self.Gravity = 1


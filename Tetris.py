import time

import numpy as np
import random
import pygame


class Tetromino:
    def __init__(self, tetromino_type="T", main_position=np.array([0, 0]), rotation=0):
        self.Type = tetromino_type
        self.Position = main_position
        self.Rotation = rotation
        self.main_piece = None
        self.sub_piece1 = None
        self.sub_piece2 = None
        self.sub_piece3 = None
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
        for r in range(self.Rotation):
            self.GetRotatedPositions()

    def GetPiecesLocation(self, main_position):
        return [main_position + self.main_piece, main_position + self.sub_piece1, main_position + self.sub_piece2,
                main_position + self.sub_piece3]


    def GetRotatedPositions(self):

        placeholder_y = self.sub_piece1[1]
        self.sub_piece1[1] = -self.sub_piece1[0]
        self.sub_piece1[0] = placeholder_y

        placeholder_y = self.sub_piece2[1]
        self.sub_piece2[1] = -self.sub_piece2[0]
        self.sub_piece2[0] = placeholder_y

        placeholder_y = self.sub_piece3[1]
        self.sub_piece3[1] = -self.sub_piece3[0]
        self.sub_piece3[0] = placeholder_y

        return self.GetPiecesLocation(self.Position)

    def RevertRotation(self):

        placeholder_y = self.sub_piece1[1]
        self.sub_piece1[1] = self.sub_piece1[0]
        self.sub_piece1[0] = -placeholder_y

        placeholder_y = self.sub_piece2[1]
        self.sub_piece2[1] = self.sub_piece2[0]
        self.sub_piece2[0] = -placeholder_y

        placeholder_y = self.sub_piece3[1]
        self.sub_piece3[1] = self.sub_piece3[0]
        self.sub_piece3[0] = -placeholder_y



class TetrisGameInstance:
    def __init__(self):
        self.Gravity = 1
        self.BufferedInput = np.array([])
        self.DefaultPosition = np.array([5, 19])
        self.PlayingGround = np.zeros((20, 10), dtype=int)
        self.PlayerPlayingGround = np.zeros((20, 10), dtype=int)
        self.TetrominoTypes = ["I", "L", "S", "Z", "O", "J", "T"]
        self.GameSpawnSeed = self.TetrominoTypes
        self.TetrominoRotationBases = {"I" : 2, "L" : 4, "S" : 2, "Z" : 2, "O" : 1, "J" : 4, "T" : 4}
        self.MyTetromino = Tetromino(tetromino_type = self.TetrominoTypes[random.randint(0, 6)], main_position = self.DefaultPosition, rotation = 0)
        self.TetrominoCounter = 0
        self.GameScreen = None
        self.TetrominoColor = None
        self.EmptySpaceColor = None
        self.LinesCleared = 0


    def getStates(self, state, piece): #canli oynanan kaydi etkilemeyen generate state fonksiyonu,zaten cnn kullanacak sadece o yuzden canliya mudahale etmeisnde sorun yok gibi
        states_to_return = []

        livePlayingGround = self.PlayingGround
        livePlayerPlayingGround = self.PlayerPlayingGround
        self.PlayerPlayingGround = np.zeros((20, 10), dtype=int)
        self.PlayingGround = state
        for r in range(self.TetrominoRotationBases[piece]):
            for i in range(1, 11):
                last_loc = np.array([])
                for j in range(19, -1, -1):
                    tetromino = Tetromino(tetromino_type=piece, main_position=np.array([i, j]),
                                      rotation=r)
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

        self.PlayingGround = livePlayingGround
        self.PlayerPlayingGround = livePlayerPlayingGround
        return states_to_return



    def ConstructGUI(self):
        self.GameScreen = pygame.display.set_mode((400, 800))
        self.TetrominoColor = (0, 0, 0)
        self.EmptySpaceColor = (255, 255, 255)

    def DrawToPlayingGround(self, positions_to_draw):
        for position in positions_to_draw:
            self.PlayingGround[(20 - position[1], position[0] - 1)] = 1

    def EraseFromPlayingGround(self, positions_to_erase):
        for position in positions_to_erase:
            self.PlayingGround[(20 - position[1], position[0] - 1)] = 0

    def DrawToPlayerPlayingGround(self, positions_to_draw):
        for position in positions_to_draw:
            self.PlayerPlayingGround[(20 - position[1], position[0] - 1)] = 1

    def EraseFromPlayerPlayingGround(self, positions_to_erase):
        for position in positions_to_erase:
            self.PlayerPlayingGround[(20 - position[1], position[0] - 1)] = 0

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

        if self.bCheckCollisionAtPosition(self.MyTetromino.GetRotatedPositions()):
            self.MyTetromino.RevertRotation()

        self.DrawToPlayerPlayingGround(self.MyTetromino.GetPiecesLocation(self.MyTetromino.Position))
        self.GameLoop()


    def BringDownLines(self, clear_row):
        for i in range(clear_row, 20):
            self.PlayingGround[20 - i] = self.PlayingGround[20 - (i + 1)] # np.all dan filtre yapilabilir, numpy c ile islme yaptigi icin daha hizli checkler ve siler ama ne kadar gerekli bilmiyorum
        self.EraseFromPlayingGround([column + 1, 20] for column in range(10))

    def CheckLineClears(self, rows):
        ClearedLineCount = 0
        for row in rows:
            if self.PlayingGround[20 - (row - ClearedLineCount)].sum() == 10:
                self.EraseFromPlayingGround([column + 1, row - ClearedLineCount] for column in range(10))
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
        canvas = self.PlayingGround
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
            self.PlayingGround = np.zeros_like(state)
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

import keyboard
import numpy as np
import random
import threading


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

    def GetPiecesLocation(self, main_position):
        return [main_position + self.main_piece, main_position + self.sub_piece1, main_position + self.sub_piece2,
                main_position + self.sub_piece3]


    def Rotate(self):

        placeholder_y = self.sub_piece1[1]
        self.sub_piece1[1] = -self.sub_piece1[0]
        self.sub_piece1[0] = placeholder_y

        placeholder_y = self.sub_piece2[1]
        self.sub_piece2[1] = -self.sub_piece2[0]
        self.sub_piece2[0] = placeholder_y

        placeholder_y = self.sub_piece3[1]
        self.sub_piece3[1] = -self.sub_piece3[0]
        self.sub_piece3[0] = placeholder_y



class TetrisGameInstance:
    def __init__(self):
        self.Gravity = 1
        self.BufferedInput = np.array([])
        self.DefaultPosition = np.array([5, 19])
        self.PlayingGround = np.zeros((20, 10), dtype=int)
        self.PlayerPlayingGround = np.zeros((20, 10), dtype=int)
        self.TetrominoTypes = ["I", "L", "S", "Z", "O", "J", "T"]
        self.MyTetromino = Tetromino(tetromino_type = self.TetrominoTypes[random.randint(0, 6)], main_position = self.DefaultPosition, rotation = 0)
        self.StartGame()

    def DrawToPlayingGround(self, positions_to_draw):
        for position in positions_to_draw:
            self.PlayingGround[(20 - position[1], position[0] - 1)] = 1

    def EraseFromPlayingGround(self, positions_to_draw):
        for position in positions_to_draw:
            self.PlayingGround[(20 - position[1], position[0] - 1)] = 0

    def DrawToPlayerPlayingGround(self, positions_to_draw):
        for position in positions_to_draw:
            self.PlayerPlayingGround[(20 - position[1], position[0] - 1)] = 1

    def EraseFromPlayerPlayingGround(self, positions_to_draw):
        for position in positions_to_draw:
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

        self.MyTetromino = Tetromino(tetromino_type= self.TetrominoTypes[random.randint(0, 6)], main_position=np.array([5, 19]), rotation=0)

        if self.bCheckCollisionAtPosition(self.MyTetromino.GetPiecesLocation(self.MyTetromino.Position)):
            return False
        else:
            self.DrawToPlayerPlayingGround(self.MyTetromino.GetPiecesLocation(self.MyTetromino.Position))
            return True

    def MoveRightInput(self, keyboard_event):
        self.BufferedInput = np.array([self.MyTetromino.Position[0] + 1, self.MyTetromino.Position[1]])

    def MoveLeftInput(self, keyboard_event):

        self.BufferedInput = np.array([self.MyTetromino.Position[0] - 1, self.MyTetromino.Position[1]])

    def MoveDownInput(self, keyboard_event):

        self.BufferedInput = np.array([self.MyTetromino.Position[0], self.MyTetromino.Position[1] - 1])


    def GameLoop(self):

        print(self.PlayingGround)

        if self.BufferedInput.size > 0:
            self.UpdateControlledTetrominoPosition(self.BufferedInput)
            self.BufferedInput = np.array([])

        if self.UpdateControlledTetrominoPosition(np.array([self.MyTetromino.Position[0], self.MyTetromino.Position[1] - self.Gravity])):
            pass
        else:
            self.DrawToPlayingGround(self.MyTetromino.GetPiecesLocation(self.MyTetromino.Position))
            if self.bTrySpawnTetromino():
                pass
            else:
                print("Game Over")
                return

        threading.Timer(0.5, self.GameLoop).start()

    def StartGame(self):

        threading.Timer(0.5, self.GameLoop).start()

















def PlayGame(input_scheme = "Keyboard"):

    GameInstance = TetrisGameInstance()

    if input_scheme == "Keyboard":
        keyboard.on_press_key('d', GameInstance.MoveRightInput)
        keyboard.on_press_key('a', GameInstance.MoveLeftInput)
        keyboard.on_press_key('s', GameInstance.MoveDownInput)

    return GameInstance



GameInstance = PlayGame("Keyboard")





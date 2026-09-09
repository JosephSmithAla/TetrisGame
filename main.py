import keyboard
import numpy as np
import random
import time

TetrominoTypes = ["I", "L", "S", "Z", "O", "J", "T"]


PlayingGround = np.zeros((20,10), dtype=int)

Gravity = 1
BufferedInput = np.array([])

def DrawToPlayingGround(positions_to_draw):
    for position in positions_to_draw:
        PlayingGround[(20 - position[1], position[0] - 1)] = 1

def EraseFromPlayingGround(positions_to_draw):
    for position in positions_to_draw:
        PlayingGround[(20 - position[1], position[0] - 1)] = 0



class Tetromino:
    def __init__(self, tetromino_type = "T", main_position = np.array([0, 0]), rotation = 0):
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
                self.main_piece = np.array([0,0])
                self.sub_piece1 = np.array([0,1])
                self.sub_piece2 = np.array([0,-1])
                self.sub_piece3 = np.array([1,-1])
            case "S":
                self.main_piece = np.array([0,0])
                self.sub_piece1 = np.array([1,0])
                self.sub_piece2 = np.array([0,-1])
                self.sub_piece3 = np.array([-1,-1])
            case "Z":
                self.main_piece = np.array([0,0])
                self.sub_piece1 = np.array([-1,0])
                self.sub_piece2 = np.array([0,-1])
                self.sub_piece3 = np.array([1,-1])
            case "O":
                self.main_piece = np.array([0,0])
                self.sub_piece1 = np.array([1,0])
                self.sub_piece2 = np.array([0,-1])
                self.sub_piece3 = np.array([1,-1])
            case "J":
                self.main_piece = np.array([0,0])
                self.sub_piece1 = np.array([0,1])
                self.sub_piece2 = np.array([0,-1])
                self.sub_piece3 = np.array([-1,-1])
            case "T":
                self.main_piece = np.array([0,0])
                self.sub_piece1 = np.array([1,0])
                self.sub_piece2 = np.array([-1,0])
                self.sub_piece3 = np.array([0,-1])

    def bCheckCollisionAtPosition(self, locations, bIgnoreSelf = True):
        if bIgnoreSelf:
            EraseFromPlayingGround(self.GetPiecesLocation(self.Position))
        for location in locations:
            if location[0] < 1 or location[0] > 10 or location[1] < 1 or location[1] > 20:
                if bIgnoreSelf:
                    DrawToPlayingGround(self.GetPiecesLocation(self.Position))
                return True
            elif PlayingGround[(20 - location[1], location[0] - 1)] == 1:
                if bIgnoreSelf:
                    DrawToPlayingGround(self.GetPiecesLocation(self.Position))
                return True
        if bIgnoreSelf:
            DrawToPlayingGround(self.GetPiecesLocation(self.Position))
        return False

    def GetPiecesLocation(self, main_position):
        return [main_position + self.main_piece, main_position + self.sub_piece1, main_position + self.sub_piece2, main_position + self.sub_piece3]

    def GetAvailablePiecesLocation(self, new_locations, old_locations):
            if self.bCheckCollisionAtPosition(new_locations):
                return old_locations
            else:
                return new_locations



    def Move(self, NewLocation):
        available_locations = self.GetAvailablePiecesLocation(self.GetPiecesLocation(NewLocation), self.GetPiecesLocation(self.Position))
        EraseFromPlayingGround(self.GetPiecesLocation(self.Position))
        self.Position = available_locations[0]
        DrawToPlayingGround(self.GetPiecesLocation(self.Position))

    def Rotate(self):

        EraseFromPlayingGround(self.GetPiecesLocation(self.Position))

        placeholder_y = self.sub_piece1[1]
        self.sub_piece1[1] = -self.sub_piece1[0]
        self.sub_piece1[0] = placeholder_y

        placeholder_y = self.sub_piece2[1]
        self.sub_piece2[1] = -self.sub_piece2[0]
        self.sub_piece2[0] = placeholder_y

        placeholder_y = self.sub_piece3[1]
        self.sub_piece3[1] = -self.sub_piece3[0]
        self.sub_piece3[0] = placeholder_y

        DrawToPlayingGround(self.GetPiecesLocation(self.Position))


MyTetromino = Tetromino(tetromino_type = TetrominoTypes[random.randint(0, 6)], main_position = np.array([5, 19]), rotation = 0)





def MoveRight(keyboard_event):
    global BufferedInput
    BufferedInput = np.array([MyTetromino.Position[0] + 1, MyTetromino.Position[1]])

def MoveLeft(keyboard_event):
    global BufferedInput
    BufferedInput = np.array([MyTetromino.Position[0] - 1, MyTetromino.Position[1]])

def MoveDown(keyboard_event):
    global BufferedInput
    BufferedInput = np.array([MyTetromino.Position[0], MyTetromino.Position[1] - 1])



keyboard.on_press_key('d', MoveRight)
keyboard.on_press_key('a', MoveLeft)
keyboard.on_press_key('s', MoveDown)

def bTrySpawnTetromino():
    global MyTetromino
    MyTetromino = Tetromino(tetromino_type = TetrominoTypes[random.randint(0, 6)], main_position = np.array([5, 19]), rotation = 0)

    return not MyTetromino.bCheckCollisionAtPosition(MyTetromino.GetPiecesLocation(MyTetromino.Position), False)


while True:

    print(PlayingGround)

    if BufferedInput.size > 0:
        MyTetromino.Move(BufferedInput)
        BufferedInput = np.array([])

    if MyTetromino.bCheckCollisionAtPosition(MyTetromino.GetPiecesLocation(np.array([MyTetromino.Position[0], MyTetromino.Position[1] - Gravity]))):
        if bTrySpawnTetromino():
            pass
        else:
            print("Game Over")
            break
    else:
        MyTetromino.Move(np.array([MyTetromino.Position[0], MyTetromino.Position[1] - Gravity]))


    time.sleep(0.5)






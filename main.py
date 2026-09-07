import keyboard
import numpy as np

TetrominoTypes = ["I", "L", "S", "Z", "O", "J", "T"]

PlayingGround = np.zeros((20,10), dtype=int)


def DrawToPlayingGround(positions_to_draw):
    global PlayingGround
    for position in positions_to_draw:
        PlayingGround[(20 - position[1], position[0] - 1)] = 1

def EraseFromPlayingGround(positions_to_draw):
    global PlayingGround
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
        DrawToPlayingGround(self.GetPiecesLocation())

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

    def GetPiecesLocation(self):
        return [self.Position + self.main_piece, self.Position + self.sub_piece1, self.Position + self.sub_piece2, self.Position + self.sub_piece3]

    def Move(self, NewLocation):
        EraseFromPlayingGround(self.GetPiecesLocation())
        self.Position = NewLocation
        DrawToPlayingGround(self.GetPiecesLocation())

    def Rotate(self):

        EraseFromPlayingGround(self.GetPiecesLocation())

        placeholder_y = self.sub_piece1[1]
        self.sub_piece1[1] = -self.sub_piece1[0]
        self.sub_piece1[0] = placeholder_y

        placeholder_y = self.sub_piece2[1]
        self.sub_piece2[1] = -self.sub_piece2[0]
        self.sub_piece2[0] = placeholder_y

        placeholder_y = self.sub_piece3[1]
        self.sub_piece3[1] = -self.sub_piece3[0]
        self.sub_piece3[0] = placeholder_y

        DrawToPlayingGround(self.GetPiecesLocation())



MyTetromino = Tetromino(tetromino_type = "I", main_position = np.array([5, 7]), rotation = 0)
print(PlayingGround)

MyTetromino.Move(np.array([3, 14]))
print(PlayingGround)

MyTetromino.Rotate()
print(PlayingGround)






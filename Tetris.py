import time
import numpy as np
import random
import pygame
from llvmlite.ir import Aggregate


class Tetromino:
    def __init__(self, tetromino_type="T", main_position=np.array([0, 0])):
        self.Type = tetromino_type
        self.Position = main_position
        self.Rotation = 0
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
        self.TetrominoMetaData = self.GenerateRotationMeta()
        self.main_piece : int
        self.sub_piece1 : int
        self.sub_piece2 : int
        self.sub_piece3 : int
        self.Construct()


    def GenerateRotationMeta(self):
        meta_data = {}
        for name, rots in self.TETROMINO_SHAPES.items():
            meta_data[name] = []
            for r in rots:
                min_x = min(dx for dx, dy in r)
                max_x = max(dx for dx, dy in r) # tetrominolarin tum rotasyonalri icin ekrandaki en sol ve en sag x degeri

                bottoms_dict = {}
                for dx, dy in r: # yere degen parcalarin koordinatlari
                    if dx not in bottoms_dict or dy < bottoms_dict[dx]:
                        bottoms_dict[dx] = dy
                bottoms = [(dx, dy) for dx, dy in bottoms_dict.items()]

                meta_data[name].append({
                    "min_x": min_x,
                    "max_x": max_x,
                    "bottoms": bottoms
                })
        return meta_data

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

        self.main_piece, self.sub_piece1, self.sub_piece2, self.sub_piece3 = np.array(self.GetRotatedPositions(rotate_count)) - self.Position

        self.Rotation += rotate_count



class TetrisGameInstance:
    def __init__(self):
        self.Gravity = 1
        self.BufferedInput = np.array([])
        self.DefaultPosition = np.array([5, 19])
        self.PlayingGround = np.zeros((20, 10), dtype=int)
        self.PlayerPlayingGround = np.zeros((20, 10), dtype=int)
        self.TetrominoTypes = ["I", "L", "S", "Z", "O", "J", "T"]
        self.CurrentShuffle = ["I", "L", "S", "Z", "O", "J", "T"]
        self.GameSpawnSeed = self.TetrominoTypes
        self.MyTetromino = None
        self.TetrominoCounter = 0
        self.GameScreen = None
        self.TetrominoColor = None
        self.EmptySpaceColor = None
        self.LinesCleared = 0
        self.CurrentStates  =np.array([])
        self.CurrentClearedLineCounts = np.array([])

    def GetProfile(self,state):
        has_blocks = np.any(state != 0, axis=0)
        col_heights = np.where(has_blocks, 20 - np.argmax(state != 0 , 0), 0)
        return col_heights


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
        if self.TetrominoCounter % 7 == 0:
            random.shuffle(self.CurrentShuffle)
        self.MyTetromino = Tetromino(tetromino_type= self.CurrentShuffle[self.TetrominoCounter % 7], main_position=np.array([5, 19]))
        self.TetrominoCounter += 1

        if self.bCheckCollisionAtPosition(self.MyTetromino.GetPiecesLocation(self.MyTetromino.Position)):
            return False
        else:
            self.DrawToPlayerPlayingGround(self.MyTetromino.GetPiecesLocation(self.MyTetromino.Position))
            return True


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

    def getStates(self, state, piece):
        livePlayingGround = self.PlayingGround.copy()
        self.PlayingGround = state.copy()
        states_to_return = []
        liveLinesCleared = self.LinesCleared

        tetromino = Tetromino(tetromino_type=piece)
        meta = tetromino.TetrominoMetaData[piece]
        col_heights = self.GetProfile(state)

        for r_id, rot_data in enumerate(meta):
            for i in range(1 - rot_data["min_x"], 10 - rot_data["max_x"] + 1):
                y_center = max([col_heights[i + dx - 1] + 1 - dy for dx, dy in rot_data["bottoms"]])
                tetromino.Position = np.array([i, y_center])
                last_loc = tetromino.GetRotatedPositions(r_id)

                max_y_in_piece = max(loc[1] for loc in last_loc) # parcanin en ustu tahtayi asiyor mu?
                if max_y_in_piece > 20:
                    continue

                cpy = self.PlayingGround.copy()
                self.DrawToPlayingGround(last_loc)
                cleared_lines = self.CheckLineClears(np.unique(np.array(last_loc)[:, 1]))

                states_to_return.append((self.PlayingGround.copy(), cleared_lines))

                self.PlayingGround = cpy
        self.PlayingGround = livePlayingGround
        self.LinesCleared = liveLinesCleared
        return states_to_return


    def getStatesOld(self, state, piece): #canli oynanan kaydi etkilemeyen generate state fonksiyonu, (state np.array((20,10)), cleared_lines int)
        states_to_return = []

        livePlayingGround = self.PlayingGround.copy()
        liveLinesCleared = self.LinesCleared
        self.PlayingGround = state.copy()

        tetromino = Tetromino(tetromino_type=piece, main_position=np.array([5, 19]))
        for r in range(tetromino.RotationMaxCount):
            for i in range(1, 10):
                last_loc = np.array([])
                for j in range(19, -1, -1):
                    tetromino.Position = np.array([i, j])
                    rot_locations = tetromino.GetRotatedPositions(r)
                    if self.bCheckCollisionAtPosition(rot_locations):
                        break
                    last_loc = rot_locations
                if len(last_loc) > 0:
                    cpy = self.PlayingGround.copy()
                    self.DrawToPlayingGround(last_loc)
                    cleared_lines = self.CheckLineClears(np.unique(np.array(last_loc)[:, 1]))
                    states_to_return.append((self.PlayingGround, cleared_lines))
                    self.PlayingGround = cpy

        self.PlayingGround = livePlayingGround.copy()
        self.LinesCleared = liveLinesCleared
        return states_to_return

    def GameLoopCNN(self, state, lines, gui = False):
        if lines < 0:
            self.StartGame()
            return state, lines, self.MyTetromino.Type, True
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
            return state, lines, self.MyTetromino.Type, False
        else:
            self.StartGame()
            return state, -100, None, True # hiper parametre olmali

    def GetMaxHeight(self, state):

        for i in range(20):
            if state[i].sum() > 0:
                return 20 - i

        return 0

    def GetHoleCount(self, state):

        col_heights = self.GetProfile(state)

        return int(np.sum(col_heights - np.sum(state, axis=0)))

    def GetBumpiness(self, state):

        heights = self.GetProfile(state)

        bumpiness = 0

        for i in range(1, 10):
            bumpiness += abs(heights[i] - heights[i - 1])

        return bumpiness

    def FeatureExtract(self, state: np.ndarray, cleared_lines: int):
        col_heights = self.GetProfile(state)
        agg_height = col_heights.sum()
        max_height = col_heights.max()
        holes = int(agg_height - state.sum())
        bumpiness = int(np.sum(np.abs(np.diff(col_heights))))

        return np.array([agg_height, max_height, cleared_lines, holes, bumpiness])


    def GetStatesNEM(self):

        LivePlayerCanvas = self.PlayerPlayingGround.copy()
        LiveCanvas = self.PlayingGround.copy()
        LiveLinesCleared = self.LinesCleared

        meta = self.MyTetromino.TetrominoMetaData[self.MyTetromino.Type]
        col_heights = self.GetProfile(self.PlayingGround)

        states = []
        Feature_Set = []
        LineClears = []

        for r_id, rot_data in enumerate(meta):
            for i in range(1 - rot_data["min_x"], 10 - rot_data["max_x"] + 1):
                y_center = max([col_heights[i + dx - 1] + 1 - dy for dx, dy in rot_data["bottoms"]])
                self.MyTetromino.Position = np.array([i, y_center])
                last_loc = self.MyTetromino.GetRotatedPositions(r_id)

                max_y_in_piece = max(loc[1] for loc in last_loc)  # parcanin en ustu tahtayi asiyor mu?
                if max_y_in_piece > 20:
                    continue

                self.DrawToPlayingGround(last_loc)
                piece_locations_y = np.unique(np.array(last_loc)[:, 1])
                ClearedLines = self.CheckLineClears(piece_locations_y)

                states.append(self.PlayingGround.copy())
                LineClears.append(ClearedLines)
                Feature_Set.append(self.FeatureExtract(self.PlayingGround.copy(), ClearedLines))
                self.PlayingGround = LiveCanvas.copy()
                self.PlayerPlayingGround = LivePlayerCanvas.copy()
                self.LinesCleared = LiveLinesCleared

        self.CurrentStates = np.array(states)
        self.CurrentClearedLineCounts = np.array(LineClears)

        return np.array(Feature_Set)




    def GameLoop(self, Chosen_State_Index):

        #for event in pygame.event.get():
            #if event.type == pygame.QUIT:
                #pygame.quit()

        #self.DrawGUI()

        self.PlayingGround = self.CurrentStates[Chosen_State_Index]
        self.LinesCleared += self.CurrentClearedLineCounts[Chosen_State_Index]

        if self.bTrySpawnTetromino():
            return 0, self.LinesCleared
        else:
            return -1 , self.LinesCleared



    def StartGame(self):

        self.PlayingGround = np.zeros((20, 10), dtype=int)
        self.PlayerPlayingGround = np.zeros((20, 10), dtype=int)
        random.shuffle(self.CurrentShuffle)
        self.MyTetromino = Tetromino(tetromino_type=self.CurrentShuffle[0], main_position=self.DefaultPosition)
        self.TetrominoCounter = 1
        self.LinesCleared = 0
        self.Gravity = 1

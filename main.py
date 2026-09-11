import keyboard
from Tetris import TetrisGameInstance

def PlayGame(input_scheme = "Keyboard"):

    GameInstance = TetrisGameInstance()

    if input_scheme == "Keyboard":
        keyboard.on_press_key('d', GameInstance.MoveRightInput)
        keyboard.on_press_key('a', GameInstance.MoveLeftInput)
        keyboard.on_press_key('s', GameInstance.MoveDownInput)
        keyboard.on_press_key('w', GameInstance.RotateInput)

    return GameInstance



GameInstance = PlayGame("Keyboard")







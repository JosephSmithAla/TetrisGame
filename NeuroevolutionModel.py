import numpy as np
from multipledispatch import dispatch
import Tetris
from blinker import signal
import time
import random





class NeuralNetwork:

    @dispatch(int, int, int)
    def __init__(self,input_count, lay1, lay2 = 4):
        self.Layer_1_weights, self.Layer_1_biases =self.InitiateLayer(lay1, input_count)
        self.Layer_2_weights, self.Layer_2_biases =self.InitiateLayer(lay2, lay1)
        self.fitness = 0
        self.LoopResult = 0
        self.LinesCleared = 0
        self.GameInstance : Tetris.TetrisGameInstance
        self.death_dispatcher = signal("death_dispatcher")


    @dispatch(np.ndarray, np.ndarray, np.ndarray, np.ndarray)
    def __init__(self, layer_1_biases, layer_1_weights, layer_2_biases, layer_2_weights):
        self.Layer_1_weights, self.Layer_1_biases = layer_1_weights, layer_1_biases
        self.Layer_2_weights, self.Layer_2_biases = layer_2_weights, layer_2_biases
        self.fitness = 0
        self.LoopResult = 0
        self.LinesCleared = 0
        self.GameInstance : Tetris.TetrisGameInstance
        self.death_dispatcher = signal("death_dispatcher")


    def FeedForward(self, input = np.array([])):

        Z1 = np.dot(self.Layer_1_weights, input) + self.Layer_1_biases

        Z2 = np.dot(self.Layer_2_weights, np.maximum(0, Z1)) + self.Layer_2_biases

        return Z2



    def InitiateLayer(self, perceptron_count, layer_input_count):

        weights = 2 * np.random.rand(perceptron_count, layer_input_count) - 1

        bias = 2 * np.random.rand(perceptron_count) - 1

        return weights,bias



    def SaveModel(self):

        np.savez("weights_and_biases.npz", lyr1b = self.Layer_1_biases, lyr1w = self.Layer_1_weights, lyr2b = self.Layer_2_biases, lyr2w = self.Layer_2_weights)


    def GetModelMoveInput(self, input):
        output = self.FeedForward(input)

        return np.argmax(output)

    def GameLoop(self):

        while self.LoopResult == 0:
            match self.GetModelMoveInput(self.GameInstance.GetGameCanvasArray()):

                case 0:
                    self.GameInstance.MoveDownInput()
                    self.fitness +=1
                case 1:
                    self.GameInstance.MoveRightInput()
                    self.fitness += 1
                case 2:
                    self.GameInstance.MoveLeftInput()
                    self.fitness += 1
                case 3:
                    self.GameInstance.RotateInput()

            self.LoopResult, self.LinesCleared = self.GameInstance.GameLoop()
        self.fitness += self.LinesCleared * 100000



        self.death_dispatcher.send(self)

    def PlayTetris(self):
        self.fitness = 0
        self.LoopResult = 0
        self.LinesCleared = 0
        self.GameInstance = Tetris.TetrisGameInstance()
        self.GameInstance.StartGame()
        self.GameLoop()



class ModelManager:
    def __init__(self, population_size):
        self.population_size = population_size
        self.Population = np.array([])
        self.death_counter = 0
        self.bShouldGenerateNextGeneration = False
        self.Mutation_Chance = 0.05
        self.Mutation_Rate = 0.1

    def CreatePopulation(self):
        arr = []
        for _ in range(self.population_size):
            NN = NeuralNetwork(200, 800, 4)
            NN.death_dispatcher.connect(self.OnNetworkDeath)
            arr.append(NN)
        self.Population = np.array(arr)
    def GenerateNewPopulation(self):

        new_generation = []
        fitness_arr = [network.fitness for network in self.Population]
        sorted_indices = np.argsort(fitness_arr)[::-1]
        self.Population = self.Population[sorted_indices]
        print(self.Population[0].fitness)

        new_generation += self.Population[0 : int(self.population_size * 5 / 100)].tolist()

        sum_fitness = sum(fitness_arr)
        normalised_fitness_weights = np.array(fitness_arr) / sum_fitness
        rng = np.random.default_rng()



        new_parents = rng.choice(self.Population, p=normalised_fitness_weights, size=int(self.population_size  / 100 * 95))

        self.Population = np.concatenate((np.array(new_generation), self.CrossParents(new_parents)))
        self.bShouldGenerateNextGeneration = False
        self.death_counter = 0

    def CrossParents(self, parents : np.ndarray):
        arr = []
        for i in range(0, parents.size, 2):
            for _ in range(2):

                childBias1 = self.CrossoverAndMutate(parents[i].Layer_1_biases, parents[i + 1].Layer_1_biases)
                childWeight1 = self.CrossoverAndMutate(parents[i].Layer_1_weights, parents[i + 1].Layer_1_weights)
                childBias2 = self.CrossoverAndMutate(parents[i].Layer_2_biases, parents[i + 1].Layer_2_biases)
                childWeight2 = self.CrossoverAndMutate(parents[i].Layer_2_weights, parents[i + 1].Layer_2_weights)

                NN = NeuralNetwork(childBias1, childWeight1, childBias2, childWeight2)
                NN.death_dispatcher.connect(self.OnNetworkDeath)
                arr.append(NN)

        return np.array(arr)

    def CrossoverAndMutate(self, p1_arr : np.ndarray, p2_arr : np.ndarray):

        mask = np.random.rand(*p1_arr.shape) < 0.5
        crossed = np.where(mask, p1_arr, p2_arr)

        probs = [1 - self.Mutation_Chance, self.Mutation_Chance / 2, self.Mutation_Chance / 2]
        deltas = np.random.choice([0, 1, -1], p=probs, size=p1_arr.shape)

        mutated = crossed + deltas * self.Mutation_Rate

        return np.clip(mutated, -1, 1)


    def TrainPopulation(self, cycle_count):
        self.CreatePopulation()
        for _ in range(cycle_count):
            print("playing")
            self.ModelsPlay()
            #while not self.bShouldGenerateNextGeneration:
                #time.sleep(1)
            print("breeding")
            self.GenerateNewPopulation()
        print("Over")

    def ModelsPlay(self):
        for Network in self.Population:
            Network.PlayTetris()


    def OnNetworkDeath(self, sender, **kwargs):
        self.death_counter += 1
        if self.death_counter >= self.population_size:
            self.bShouldGenerateNextGeneration = True


Manager = ModelManager(200)
Manager.TrainPopulation(10000)
print("Training Over")
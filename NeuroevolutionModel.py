import numpy as np
import Tetris
from concurrent.futures import ProcessPoolExecutor
import os
from numba import njit



class NeuralNetwork:


    def CreateLayers(input_count, lay1_count, lay2_count):

        lay1_weights = 2 * np.random.rand(lay1_count, input_count) - 1
        lay1_biases = 2 * np.random.rand(lay1_count) - 1
        lay2_weights = 2 * np.random.rand(lay2_count, lay1_count) - 1
        lay2_biases = 2 * np.random.rand(lay2_count) - 1

        return [lay1_weights, lay1_biases, lay2_weights, lay2_biases]

    @njit(fastmath=True)
    def FeedForward(input_arr = np.array([]), w1 = [], b1 = [], w2 = [], b2 = []):

        input_arr = input_arr.astype(np.float64)

        Z1 = np.dot(w1, input_arr) + b1
        Z2 = np.dot(w2, np.maximum(0, Z1)) + b2

        return Z2

    def GetModelMoveInput(input_arr, weights_and_biases):
        output = NeuralNetwork.FeedForward(input_arr, *weights_and_biases)

        return np.argmax(output)

    def GameLoop(weights_and_biases):

        GameInstance = Tetris.TetrisGameInstance()
        GameInstance.StartGame()
        LoopResult = 0
        LinesCleared = 0

        fitness = 0
        while LoopResult == 0:
            match NeuralNetwork.GetModelMoveInput(GameInstance.GetGameCanvasArray(),weights_and_biases):

                case 0:
                    GameInstance.MoveDownInput()
                    fitness +=1
                case 1:
                    GameInstance.MoveRightInput()
                    fitness += 1
                case 2:
                    GameInstance.MoveLeftInput()
                    fitness += 1
                case 3:
                    GameInstance.RotateInput()

            LoopResult, LinesCleared = GameInstance.GameLoop()
        fitness += LinesCleared * 100000
        return fitness, LinesCleared

    def PlayTetris(weights_and_biases):

        fitness, LinesCleared = NeuralNetwork.GameLoop(weights_and_biases)

        return fitness



class ModelManager:
    def __init__(self, population_size):
        self.population_size = population_size
        self.Population = []
        self.fitness_arr = np.array([])
        self.death_counter = 0
        self.bShouldGenerateNextGeneration = False
        self.Mutation_Chance = 0.05
        self.Mutation_Rate = 0.1

    def CreatePopulation(self):

        for _ in range(self.population_size):
            NN = NeuralNetwork.CreateLayers(200, 200, 4)#input - layer 1 , layer 2, NN = wbwb
            self.Population.append(NN)


    def GenerateNewPopulation(self):


        new_generation = []
        sorted_indices = np.argsort(self.fitness_arr)[::-1]
        self.Population = [self.Population[i] for i in sorted_indices]
        self.fitness_arr = self.fitness_arr[sorted_indices]
        print(self.fitness_arr[0])

        new_generation += self.Population[0 : int(self.population_size * 5 / 100)]

        sum_fitness = self.fitness_arr.sum()
        normalised_fitness_weights = self.fitness_arr / sum_fitness
        rng = np.random.default_rng()

        indices = rng.choice(len(self.Population), p=normalised_fitness_weights, size=int(self.population_size / 100 * 95))

        new_parents = [self.Population[i] for i in indices]

        self.Population = new_generation + self.CrossParents(new_parents)

        self.bShouldGenerateNextGeneration = False
        self.death_counter = 0
        self.fitness_arr = np.zeros(self.fitness_arr.shape)

    def CrossParents(self, parents : np.ndarray):
        arr = []
        for i in range(0, len(parents), 2):
            for _ in range(2):

                childWeight1 = self.CrossoverAndMutate(parents[i][0], parents[i + 1][0])
                childBias1 = self.CrossoverAndMutate(parents[i][1], parents[i + 1][1])
                childWeight2 = self.CrossoverAndMutate(parents[i][2], parents[i + 1][2])
                childBias2 = self.CrossoverAndMutate(parents[i][3], parents[i + 1][3])

                NN = [childWeight1, childBias1, childWeight2, childBias2]
                arr.append(NN)

        return arr

    def CrossoverAndMutate(self, p1_arr : np.ndarray, p2_arr : np.ndarray):

        mask = np.random.rand(*p1_arr.shape) < 0.5
        crossed = np.where(mask, p1_arr, p2_arr)

        probs = [1 - self.Mutation_Chance, self.Mutation_Chance / 2, self.Mutation_Chance / 2]
        deltas = np.random.choice([0, 1, -1], p=probs, size=p1_arr.shape)

        mutated = crossed + deltas * self.Mutation_Rate

        return np.clip(mutated, -1, 1)

    def TrainPopulation(self, cycle_count):
        self.CreatePopulation()

        cpu_count = os.cpu_count() or 4

        chunk_size = max(1, self.population_size // (cpu_count * 2))

        with ProcessPoolExecutor(max_workers=cpu_count) as Executor:
            for _ in range(cycle_count):

                self.fitness_arr = np.array(list(Executor.map(NeuralNetwork.PlayTetris, self.Population, chunksize=chunk_size)))

                self.GenerateNewPopulation()


        print("Over")

    def ModelsPlay(self):

        for Network in self.Population:
            Network.PlayTetris()

if __name__ == "__main__":
    Manager = ModelManager(200)
    Manager.TrainPopulation(10000)
    print("Training Over")


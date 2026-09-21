from abc import abstractclassmethod
from typing import List

import numpy as np
from multipledispatch import dispatch
import Tetris
import time
import random
from concurrent.futures import ProcessPoolExecutor



class NeuralNetwork:


    def CreateLayers(input_count, lay1_count, lay2_count):

        lay1_weights = 2 * np.random.rand(lay1_count, input_count) - 1
        lay1_biases = 2 * np.random.rand(lay1_count) - 1
        lay2_weights = 2 * np.random.rand(lay2_count, lay1_count) - 1
        lay2_biases = 2 * np.random.rand(lay1_count) - 1

        return np.array([lay1_weights, lay1_biases, lay2_weights, lay2_biases])

    def FeedForward(input = np.array([]), weights_and_biases = np.array([])):

        Z1 = np.dot(weights_and_biases[0], input) + weights_and_biases[1]
        Z2 = np.dot(weights_and_biases[2], np.maximum(0, Z1)) + weights_and_biases[3]

        return Z2

    def GetModelMoveInput(input, weights_and_biases):
        output = NeuralNetwork.FeedForward(input, weights_and_biases)

        return np.argmax(output)

    def GameLoop(weights_and_biases):

        GameInstance = Tetris.TetrisGameInstance()
        GameInstance.StartGame()

        fitness = 0
        while LoopResult == 0:
            match NeuralNetwork.GetModelMoveInput(NeuralNetwork.GameInstance.GetGameCanvasArray(),weights_and_biases):

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


    def PlayTetris(weights_and_biases):
        fitness = 0
        LinesCleared = 0

        fitness, LinesCleared = NeuralNetwork.GameLoop(weights_and_biases)

        return fitness



class ModelManager:
    def __init__(self, population_size):
        self.population_size = population_size
        self.Population = np.array([])
        self.fitnesses = np.array([])
        self.death_counter = 0
        self.bShouldGenerateNextGeneration = False
        self.Mutation_Chance = 0.05
        self.Mutation_Rate = 0.1

    def CreatePopulation(self):
        arr = []
        for _ in range(self.population_size):
            NN = NeuralNetwork.CreateLayers(400, 200, 4)#input - layer 1 , layer 2, NN = wbwb
            arr.append(NN)
        self.Population = np.array(arr)

    def GenerateNewPopulation(self):


        new_generation = []
        sorted_indices = np.argsort(self.fitnesses)[::-1]
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
            with ProcessPoolExecutor as Executor:
                fitness_results = List(Executor.map(NeuralNetwork.PlayTetris, self.Population))

            self.GenerateNewPopulation()

        print("Over")

    def ModelsPlay(self):

        for Network in self.Population:
            Network.PlayTetris()

if __name__ == "__main__":
    Manager = ModelManager(200)
    Manager.TrainPopulation(10000)
    print("Training Over")


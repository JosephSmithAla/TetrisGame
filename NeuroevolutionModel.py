import numpy as np
from multipledispatch import dispatch
import Tetris
from blinker import signal
import time
import random


class Perceptron:
    @dispatch(object, np.ndarray)
    def __init__(self, input_bias = 0, input_weights = np.array([])):

        self.weights = input_weights
        self.bias = input_bias


    @dispatch(int)
    def __init__(self, input_count = 0):

        self.weights = 2 * np.random.rand(input_count) - 1
        self.bias = 2 * np.random.rand(1)[0] - 1


    def FeedForward(self, input = np.array([])):
        output = np.dot(self.weights, input) + self.bias
        return output



class NeuralNetwork:

    @dispatch(int, int, int)
    def __init__(self,input_count, lay1, lay2 = 4):
        self.Layer_1 =self.InitiatePerceptronLayer(lay1, input_count)
        self.Layer_2 =self.InitiatePerceptronLayer(lay2, lay1)
        self.fitness = -1
        self.GameInstance : Tetris.TetrisGameInstance
        self.death_dispatcher = signal("death_dispatcher")


    @dispatch(np.ndarray, np.ndarray, np.ndarray, np.ndarray)
    def __init__(self, layer_1_biases, layer_1_weights, layer_2_biases, layer_2_weights):
        self.Layer_1 =self.LoadPerceptronLayer(layer_1_biases.size, layer_1_weights, layer_1_biases)
        self.Layer_2 =self.LoadPerceptronLayer(layer_2_biases.size, layer_2_weights, layer_2_biases)
        self.fitness = -1
        self.GameInstance : Tetris.TetrisGameInstance
        self.death_dispatcher = signal("death_dispatcher")


    def FeedForward(self, input = np.array([])):

        output_1 = []

        for i in range(len(self.Layer_1)):
            output_1.append(self.Activation_1(self.Layer_1[i].FeedForward(input)))
        output_2 = []

        for i in range(len(self.Layer_2)):
            output_2.append(self.Activation_2(self.Layer_2[i].FeedForward(output_1)))

        return output_2


    def LoadPerceptronLayer(self, perceptron_count, weights = np.array([]), biases = np.array([])):
        perceptrons = np.array([])
        for i in range(perceptron_count):
            perceptrons = np.append(perceptrons, Perceptron(biases[i], weights[i]))

        return perceptrons

    def InitiatePerceptronLayer(self, perceptron_count, layer_input_count):
        perceptrons = np.array( [])
        for i in range(perceptron_count):
            perceptrons = np.append(perceptrons, Perceptron(layer_input_count))
        return perceptrons

    def GetWeights(self, layer):

        return np.array([neuron.weights for neuron in layer])

    def GetBiases(self, layer):

        return np.array([neuron.bias for neuron in layer])

    def SaveModel(self):
        layer_1_biases = self.GetBiases(self.Layer_1)
        layer_1_weights = self.GetWeights(self.Layer_1)
        layer_2_biases = self.GetBiases(self.Layer_2)
        layer_2_weights = self.GetWeights(self.Layer_2)
        np.savez("weights_and_biases.npz", lyr1b = layer_1_biases, lyr2w = layer_1_weights, lyr2b = layer_2_biases, lyr3w = layer_2_weights)

    def Activation_1(self, output):
        if output > 0:
            return output
        else:
            return 0

    def Activation_2(self, output):
        return output

    def GetModelMoveInput(self, input):
        output = self.FeedForward(input)

        return output.index(max(output))

    def GameLoop(self):

        while self.fitness == -1:
            match self.GetModelMoveInput(self.GameInstance.GetGameCanvasArray()):

                case 0:
                    self.fitness = self.GameInstance.MoveDownInput()
                case 1:
                    self.fitness = self.GameInstance.MoveRightInput()
                case 2:
                    self.fitness = self.GameInstance.MoveLeftInput()
                case 3:
                    self.fitness = self.GameInstance.RotateInput()
            self.fitness = self.GameInstance.GameLoop()

        self.death_dispatcher.send(self)

    def PlayTetris(self):
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
        for _ in range(self.population_size):
            NN = NeuralNetwork(400, 100, 4)
            NN.death_dispatcher.connect(self.OnNetworkDeath)
            self.Population = np.append(self.Population, NN)

    def GenerateNewPopulation(self):

        new_generation = np.array([])

        sorted_indices = np.argsort([network.fitness for network in self.Population])
        self.Population = self.Population[sorted_indices][::-1]

        for i in range(int(self.population_size / 100 * 10)):
            new_generation = np.append(new_generation, self.Population[i])

        normalised_fitness_weights = np.array([network.fitness for network in self.Population]) / np.array([network.fitness for network in self.Population]).sum()
        rng = np.random.default_rng()

        new_parents = np.array([])

        for i in range(int(self.population_size - (self.population_size / 100 * 10))):
            new_parents = np.append(new_parents, rng.choice(self.Population, p=normalised_fitness_weights))

        self.Population = np.concatenate((new_generation, self.CrossParents(new_parents)))
        self.bShouldGenerateNextGeneration = False
        self.death_counter = 0

    def CrossParents(self, parents : np.ndarray):
        arr = np.array([])
        for i in range(0, parents.size, 2):
            parent_1 = [parents[i].GetBiases(parents[i].Layer_1), parents[i].GetWeights(parents[i].Layer_1), parents[i].GetBiases(parents[i].Layer_2), parents[i].GetWeights(parents[i].Layer_2)]
            parent_2 =[parents[i + 1].GetBiases(parents[i + 1].Layer_1), parents[i + 1].GetWeights(parents[i + 1].Layer_1), parents[i + 1].GetBiases(parents[i + 1].Layer_2), parents[i + 1].GetWeights(parents[i + 1].Layer_2)]
            for _ in range(2):

                childBias1 = self.GetRandomCrossedAndMutatedBiases(parent_1[0], parent_2[0])
                childWeight1 = self.GetRandomCrossedAndMutatedWeights(parent_1[1], parent_2[1])
                childBias2 = self.GetRandomCrossedAndMutatedBiases(parent_1[2], parent_2[2])
                childWeight2 = self.GetRandomCrossedAndMutatedWeights(parent_1[3], parent_2[3])

                NN = NeuralNetwork(childBias1, childWeight1, childBias2, childWeight2)
                NN.death_dispatcher.connect(self.OnNetworkDeath)
                arr = np.append(arr, NN)

        return arr

    def GetRandomCrossedAndMutatedWeights(self, p1_w, p2_w):
        new_weights = []
        arguments = [0, -1, 1]
        prob_weights = [1 - self.Mutation_Chance, self.Mutation_Chance / 2, self.Mutation_Chance / 2]

        for i in range(len(p1_w)):
            selected_weights = []
            for j in range(len(p1_w[i])):
                new_weight = random.choice([p1_w, p2_w])[i][j] + self.Mutation_Rate * random.choices(arguments, weights=prob_weights, k=1)[0]
                selected_weights.append(new_weight)
            new_weights.append(np.array(selected_weights))
        return np.array(new_weights)

    def GetRandomCrossedAndMutatedBiases(self, p1_b, p2_b):
        mutated_biases = []
        arguments = [0, -1, 1]
        weights = [1 - self.Mutation_Chance, self.Mutation_Chance / 2, self.Mutation_Chance / 2]

        for i in range(p1_b.size):
            new_bias = random.choice([p1_b, p2_b])[i] + self.Mutation_Rate * random.choices(arguments, weights=weights, k=1)[0]

            mutated_biases.append(new_bias)

        return np.array(mutated_biases)



    def MutateBias(self, biases):
        mutated_biases = np.array([])
        arguments = [0, -1, 1]
        weights = [1 - self.Mutation_Chance, self.Mutation_Chance / 2, self.Mutation_Chance / 2]

        for bias in biases:
            new_bias = bias + self.Mutation_Rate *  random.choices(arguments, weights=weights, k=1)[0]

            mutated_biases = np.append(mutated_biases, new_bias)

        return mutated_biases

    def TrainPopulation(self, cycle_count):
        self.CreatePopulation()
        for _ in range(cycle_count):
            self.ModelsPlay()
            while not self.bShouldGenerateNextGeneration:
                time.sleep(1)
            self.GenerateNewPopulation()
        sorted_indices = np.argsort([network.fitness for network in self.Population])
        self.Population = self.Population[sorted_indices][::-1]
        the_fittest = self.Population[0]
        the_fittest.SaveModel()

    def ModelsPlay(self):
        for Network in self.Population:
            Network.PlayTetris()


    def OnNetworkDeath(self, sender, **kwargs):
        self.death_counter += 1
        if self.death_counter >= self.population_size:
            self.bShouldGenerateNextGeneration = True



import numpy as np
import Tetris
import os
import torch
import torch.multiprocessing as mp

INPUT_COUNT = 200
LAYER1_COUNT = 32
LAYER2_COUNT = 4

OFFSET1 = INPUT_COUNT * LAYER1_COUNT
OFFSET2 = OFFSET1 + LAYER1_COUNT
OFFSET3 = OFFSET2 + LAYER2_COUNT * LAYER1_COUNT
TOTAL_PARAMETERS = OFFSET3 + LAYER2_COUNT
#ofst_1 birinci layerin ağırlıklar
#ofst_2 birinci layerin biaslar
#ofst_3 ikinci layerin ağırlıklar
#total_params ikinci layerin biaslar

GLOBAL_POPULATION : torch.Tensor


def init_worker(shared_population):
    global GLOBAL_POPULATION
    GLOBAL_POPULATION = shared_population

class NeuralNetwork:

    @staticmethod
    def UnpackWeightsAndBiases(flat_tensor : torch.Tensor):

        w1 = flat_tensor[:OFFSET1].view(LAYER1_COUNT, INPUT_COUNT)
        b1 = flat_tensor[OFFSET1:OFFSET2]
        w2 = flat_tensor[OFFSET2:OFFSET3].view(LAYER2_COUNT, LAYER1_COUNT)
        b2 = flat_tensor[OFFSET3:TOTAL_PARAMETERS]

        return w1, b1, w2, b2

    @staticmethod
    def CreateLayers(input_count, lay1_count, lay2_count):

        lay1_weights = 2 * np.random.rand(lay1_count, input_count) - 1
        lay1_biases = 2 * np.random.rand(lay1_count) - 1
        lay2_weights = 2 * np.random.rand(lay2_count, lay1_count) - 1
        lay2_biases = 2 * np.random.rand(lay2_count) - 1

        return [lay1_weights, lay1_biases, lay2_weights, lay2_biases]

    @staticmethod
    def FeedForward(input_tensor, w1, b1, w2, b2):



        Z1 = torch.relu(torch.matmul(w1, input_tensor) + b1)
        Z2 = torch.matmul(w2, Z1) + b2

        return Z2

    @staticmethod
    def GetModelMoveInput(input_tensor, weights_and_biases):
        output = NeuralNetwork.FeedForward(input_tensor, *weights_and_biases)

        return torch.argmax(output).item()

    @staticmethod
    def GameLoop(weights_and_biases : list):

        GameInstance = Tetris.TetrisGameInstance()
        GameInstance.StartGame()
        LoopResult = 0
        LinesCleared = 0

        fitness = 0

        with torch.no_grad():
            while LoopResult == 0:

                canvas_arr = GameInstance.GetGameCanvasArray()
                input_tensor = torch.from_numpy(canvas_arr).float().flatten()

                match NeuralNetwork.GetModelMoveInput(input_tensor, weights_and_biases):

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
            fitness += LinesCleared * 20000
            return fitness, LinesCleared

    @staticmethod
    def PlayTetris(individual_index : int):


        flat_weights = GLOBAL_POPULATION[individual_index]

        w1, b1, w2, b2 = NeuralNetwork.UnpackWeightsAndBiases(flat_weights)

        fitness, LinesCleared = NeuralNetwork.GameLoop([w1, b1, w2, b2])

        return fitness



class ModelManager:
    def __init__(self, population_size):
        self.population_size = population_size
        self.Population = torch.zeros((self.population_size, TOTAL_PARAMETERS), dtype = torch.float32)
        self.Population.share_memory_()
        self.fitness_arr = np.zeros(self.population_size)
        self.death_counter = 0
        self.bShouldGenerateNextGeneration = False
        self.Mutation_Chance = 0.05
        self.Mutation_Rate = 0.1


    def CreatePopulation(self):

        self.Population.uniform_(-1.0, 1.0)



    def GenerateNewPopulation(self):


        new_generation = torch.zeros_like(self.Population)

        sorted_indices = np.argsort(self.fitness_arr)[::-1]


        best_population = int(self.population_size * 0.1)
        for i in range(best_population):

            new_generation[i] = self.Population[sorted_indices[i]]


        print(self.fitness_arr[sorted_indices[0]])


        sum_fitness = self.fitness_arr.sum()
        normalised_fitness_weights = self.fitness_arr / sum_fitness
        rng = np.random.default_rng()

        p1_indices = rng.choice(len(self.Population), p=normalised_fitness_weights, size=int(self.population_size / 100 * 90))

        p2_indices = rng.choice(len(self.Population), p=normalised_fitness_weights, size=int(self.population_size / 100 * 90))

        new_generation[best_population:] = self.CrossParents(self.Population[p1_indices], self.Population[p2_indices])

        self.Population.copy_(new_generation)

        self.bShouldGenerateNextGeneration = False
        self.death_counter = 0
        self.fitness_arr = np.zeros(self.fitness_arr.shape)

    def CrossParents(self, parents_1, parents_2):

        crossover_mask = torch.rand_like(parents_1) < 0.5
        children = torch.where(crossover_mask, parents_1, parents_2)

        mutation_mask = torch.rand_like(children) < self.Mutation_Chance
        random_deltas = 2 * torch.rand_like(children) - 1
        children += mutation_mask * random_deltas * self.Mutation_Rate

        return torch.clamp(children, -1.0, 1.0)


    def TrainPopulation(self, cycle_count):
        self.CreatePopulation()

        cpu_count = os.cpu_count() or 4

        with mp.Pool(processes=cpu_count, initializer=init_worker, initargs= (self.Population,)) as pool:
            for _ in range(cycle_count):

                results = pool.map(NeuralNetwork.PlayTetris, range(self.population_size))

                self.fitness_arr = np.array(results)

                self.GenerateNewPopulation()


        print("Over")


if __name__ == "__main__":

    mp.set_start_method("spawn", force=True)

    Manager = ModelManager(100)
    Manager.TrainPopulation(1000000)
    print("Training Over")


import numpy as np
import Tetris
import os
import torch
import torch.multiprocessing as mp
import pygame
import sys

INPUT_COUNT = 5
LAYER1_COUNT = 32
LAYER2_COUNT = 1

POPULATION_COUNT = 100

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
        flat_np = flat_tensor.numpy()

        w1 = flat_np[:OFFSET1].reshape(LAYER1_COUNT, INPUT_COUNT)
        b1 = flat_np[OFFSET1:OFFSET2]
        w2 = flat_np[OFFSET2:OFFSET3].reshape(LAYER2_COUNT, LAYER1_COUNT)
        b2 = flat_np[OFFSET3:TOTAL_PARAMETERS]

        return w1, b1, w2, b2

    @staticmethod
    def CreateLayers(input_count, lay1_count, lay2_count):

        lay1_weights = 2 * np.random.rand(lay1_count, input_count) - 1
        lay1_biases = 2 * np.random.rand(lay1_count) - 1
        lay2_weights = 2 * np.random.rand(lay2_count, lay1_count) - 1
        lay2_biases = 2 * np.random.rand(lay2_count) - 1

        return [lay1_weights, lay1_biases, lay2_weights, lay2_biases]

    @staticmethod
    def FeedForward(features, w1, b1, w2, b2):



        Z1 = np.maximum(0, np.dot(w1, features) + b1)
        Z2 = np.dot(w2, Z1) + b2

        return Z2.item()

    @staticmethod
    def GetModelStateIndex(input_arr, weights_and_biases):

        evaluation_list = [NeuralNetwork.FeedForward(features, *weights_and_biases) for features in input_arr]

        return np.argmax(evaluation_list)

    @staticmethod
    def GameLoop(weights_and_biases : list):

        GameInstance = Tetris.TetrisGameInstance()
        GameInstance.StartGame()
        LoopResult = 0
        LinesCleared = 0

        fitness = 1

        with torch.no_grad():


            while LoopResult == 0:

                if LinesCleared >= 50000:
                    break

                feature_set = GameInstance.GetStatesNEM()

                if feature_set.size == 0:
                    break



                StateIndex =  NeuralNetwork.GetModelStateIndex(feature_set, weights_and_biases)

                LoopResult, LinesCleared = GameInstance.GameLoop(StateIndex)



            fitness += LinesCleared * POPULATION_COUNT
            return fitness, LinesCleared

    @staticmethod
    def PlayTetris(individual_index : int):


        flat_weights = GLOBAL_POPULATION[individual_index]

        w1, b1, w2, b2 = NeuralNetwork.UnpackWeightsAndBiases(flat_weights)

        fitness, LinesCleared = NeuralNetwork.GameLoop([w1, b1, w2, b2])


        return fitness

    @staticmethod
    def PlayTetrisVisual(flat_weights: torch.Tensor, fps: int = 15):
        pygame.init()

        BLOCK_SIZE = 30
        GRID_WIDTH, GRID_HEIGHT = 10, 20
        SCREEN_WIDTH = GRID_WIDTH * BLOCK_SIZE
        SCREEN_HEIGHT = GRID_HEIGHT * BLOCK_SIZE

        screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("AI Tetris - Best Individual")
        clock = pygame.time.Clock()

        pygame.event.pump()
        clock = pygame.time.Clock()

        w1, b1, w2, b2 = NeuralNetwork.UnpackWeightsAndBiases(flat_weights)
        weights_and_biases = [w1, b1, w2, b2]

        GameInstance = Tetris.TetrisGameInstance()
        GameInstance.StartGame()

        LoopResult = 0
        LinesCleared = 0

        while LoopResult == 0:

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return

            feature_set = GameInstance.GetStatesNEM()
            if feature_set.size == 0:
                break

            StateIndex = NeuralNetwork.GetModelStateIndex(feature_set, weights_and_biases)
            LoopResult, LinesCleared = GameInstance.GameLoop(StateIndex)


            screen.fill((15, 15, 20))

            board = GameInstance.PlayingGround

            if board is not None:
                for r in range(GRID_HEIGHT):
                    for c in range(GRID_WIDTH):
                        if board[r][c] != 0:
                            rect = pygame.Rect(c * BLOCK_SIZE, r * BLOCK_SIZE, BLOCK_SIZE - 1, BLOCK_SIZE - 1)
                            pygame.draw.rect(screen, (0, 230, 150), rect)

            pygame.display.flip()
            #clock.tick(fps)

        print(f"Görsel Simülasyon Bitti. Toplam Temizlenen Satır: {LinesCleared}")
        pygame.quit()



class ModelManager:
    def __init__(self, population_size):
        self.population_size = population_size
        self.Population = torch.zeros((self.population_size, TOTAL_PARAMETERS), dtype = torch.float32)
        self.Population.share_memory_()
        self.fitness_arr = np.zeros(self.population_size)
        self.death_counter = 0
        self.bShouldGenerateNextGeneration = False
        self.Mutation_Chance = 0.15
        self.Mutation_Rate = 0.4


    def CreatePopulation(self):

        self.Population.uniform_(-1.0, 1.0)



    def GenerateNewPopulation(self):


        new_generation = torch.zeros_like(self.Population)

        sorted_indices = np.argsort(self.fitness_arr)[::-1]


        best_population = int(self.population_size * 0.1)
        for i in range(best_population):

            new_generation[i] = self.Population[sorted_indices[i]]


        print(self.fitness_arr[sorted_indices])


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

        pool = mp.Pool(processes=cpu_count, initializer=init_worker, initargs=(self.Population,))
        try:
            for i in range(cycle_count):

                print("Current Generation : ", i+1)

                generation_fitness = np.zeros(self.population_size)


                num_episodes = 3
                for _ in range(num_episodes):
                    results = pool.map(NeuralNetwork.PlayTetris, range(self.population_size))
                    generation_fitness += np.array(results)

                self.fitness_arr = generation_fitness / num_episodes

                if i+1 != cycle_count:
                    self.GenerateNewPopulation()
                    self.fitness_arr = np.zeros(self.population_size)


        finally:
            pool.close()
            pool.join()
        sorted_indices = np.argsort(self.fitness_arr)[::-1]
        best_model = self.Population[sorted_indices[0]].clone()


        print("\n--- Eğitim Tamamlandı! En İyi Birey Oynatılıyor ---")
        for _ in range(10):
            NeuralNetwork.PlayTetrisVisual(best_model, fps=15)


if __name__ == "__main__":

    mp.set_start_method("spawn", force=True)

    Manager = ModelManager(POPULATION_COUNT)
    Manager.TrainPopulation(10)
    print("Training Over")


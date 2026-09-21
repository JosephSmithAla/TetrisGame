from os import name
from collections import deque
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import random
from Tetris import TetrisGameInstance

class TetrisModel: # pooling olmamasi sart cunku indirgeme yapoiyor pooling. daha cok image icin feature extractionda parametreleri azaltmak icin kullaniliyor
    def __init__(self, lr, frequency, epsilon_decrease_rate):
        #PARALLEL CONVOLUTION FILTERS
        inputs = layers.Input(shape=(20, 10, 1)) # yukseklik genislik kanal
        branch1x3 = layers.Conv2D(32, (1, 3), activation='relu', padding='same')(inputs) #yatay bilgiler
        branch5x1 = layers.Conv2D(32, (5, 1), activation='relu', padding='same')(inputs) #dikey bilgiler
        branch3x3 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs) #genel feature extraction
        x = layers.Concatenate(axis=-1)([branch1x3, branch5x1, branch3x3])
        x = layers.Conv2D(16, (1, 1), activation='relu')(x) #parametre sayisini azaltmak icin
        x = layers.Flatten()(x)
        x = layers.Dense(64, activation='relu')(x)
        output = layers.Dense(1, activation='linear')(x) # negatif cezalari isleyebilmesi icin linear

        self.model = models.Model(inputs=inputs, outputs=output, name='TetrisModel')
        self.target_model = models.Model(inputs=inputs, outputs=output, name='TargetTetrisModel')
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr) # hiper parametre olarak eklenmeli EKLENDI
        self.model.summary()

        self.game = TetrisGameInstance()
        self.model_update_frequency = frequency
        self.model_update_counter = 0 # hiper parametre olarak eklenmeli EKLENDI
        self.memory = deque(maxlen=20000) # 20x10np.array (s), int (reward), int (next_piece_enum), done
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decrease_rate = epsilon_decrease_rate

        self.lines = [0]
        self.loss_values = []

    def sample(self, sample_size):
        return random.sample(self.memory, sample_size)


    def memorize(self, state, lines, piece, done):
        if lines >= 0:
            reward = pow(2, lines) * lines * lines / 10
        else:
            reward = lines
        self.memory.append((state, reward, piece, done))

    def play(self, play_num):
        lines_sum = 0
        for i in range(play_num):
            s_r = self.game.getStates(self.game.PlayingGround, self.game.MyTetromino.Type)
            rand = random.random()
            if self.epsilon < rand: #epsilon-greedy algoritmasi
                possible_states = np.array([np.expand_dims(item[0], axis=-1) for item in s_r], dtype=np.float32) # hem iceride kanal ekliyorus axis -1 ile hem de disaridan sariyoruz boylelikle dis boyut da artiyor
                decision = s_r[np.argmax(self.model(possible_states, training=False).numpy())]
            else:
                decision = random.choice(s_r)
            if decision[1] > 0:
                lines_sum += decision[1]
            self.memorize(*self.game.GameLoopCNN(*decision))
        self.lines.append((lines_sum + self.lines[-1]))

    def save(self):
        self.model.save('CNNModel.h5')
        np.savez('CNNModel_meta.npz', loss=self.loss_values, lines=self.lines, epsilon=self.epsilon)

    def load(self, epsilon = None):
        self.model = models.load_model('CNNModel.h5')
        self.target_model.set_weights(self.model.get_weights())
        file = np.load('CNNModel_meta.npz')
        self.loss_values = file['loss'].tolist()
        self.lines = file['lines'].tolist()
        if epsilon is None:
            self.epsilon = file['epsilon']
        else:
            self.epsilon = epsilon

    def optimize(self, batch_size):
        if len(self.memory) < batch_size:
            return

        # target model guncellemesi
        self.model_update_counter += 1
        if(self.model_update_counter == self.model_update_frequency):
            self.target_model.set_weights(self.model.get_weights())
            self.model_update_counter = 0

        #epsilon azaltma
        if (self.epsilon >= self.epsilon_min):
            self.epsilon *= self.epsilon_decrease_rate
        else:
            self.epsilon = self.epsilon_min

        batch = self.sample(batch_size)
        states = np.array([np.expand_dims(item[0], axis=-1) for item in batch], dtype=np.float32) #(BATCH_SIZE, 20, 10, 1)
        rewards = np.array([item[1] for item in batch], dtype=np.float32)
        max_future_qs = np.zeros(batch_size, dtype=np.float32)
        future_states = []
        fstates_id = [] # future statelerin sirasini kaybetmemek icin
        filter_id = [] # kendi aralarinda hangi gruba ait olduklarini unutmamak icin
        id = 0
        for i in range(batch_size):
            if (not batch[i][3]):
                next_states = self.game.getStates(batch[i][0], batch[i][2])
                next_states = [r[0] for r in next_states]
                for s in next_states:
                    future_states.append(s)
                    filter_id.append(id)
                if len(next_states) > 0:
                    fstates_id.append(i)
                    id += 1

        if id > 0: #hic gelen bir sey yoksa done hepsi icin true ise bosuna gpu calismasin
            future_states = np.expand_dims(np.array(future_states, dtype=np.float32),
                                           axis=-1)  # (batch_size, 20, 10 ,1)
            q_preds = self.target_model(future_states, training=False)
            q_preds = tf.squeeze(q_preds, axis=-1)
            segment_ids = tf.constant(filter_id, dtype=tf.int32)
            max_future_qs[fstates_id] = tf.math.segment_max(q_preds, segment_ids).numpy()

        target = rewards + np.float32(0.997) * max_future_qs # bellman equation
        target = tf.expand_dims(target, axis=-1) # dimensionlari ayni tutmak icin yoksa 1 boyut eksik oluyor

        with tf.GradientTape() as tape: # forward pass kayit altina aliniyor ki backward propda gradyan bulunabilsin
            pred_states = self.model(states, training=True)
            loss = tf.keras.losses.mse(target, pred_states)
            loss = tf.reduce_mean(loss)

        grad = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grad, self.model.trainable_variables))
        self.loss_values.append(loss)

    def info(self):
        print("LOSS:", float(self.loss_values[-1]), " | ", "LINES:", self.lines[-1], " | EPSILON:", self.epsilon)

    def plot(self):
        fig, ax1 = plt.subplots()
        ax1.set_xlabel('optimize steps')
        ax1.set_ylabel('loss (log10)', color='blue')
        ax1.plot(np.log10(self.loss_values), color='blue')

        ax2 = ax1.twinx()
        ax2.set_ylabel('lines cleared', color='orange')
        ax2.plot(self.lines, color='orange')
        plt.show()

    def play_test(self, play_num):
        self.game.ConstructGUI()
        self.game.StartGame()
        for i in range(play_num):
            s_r = self.game.getStates(self.game.PlayingGround, self.game.MyTetromino.Type)
            possible_states = np.array([np.expand_dims(item[0], axis=-1) for item in s_r], dtype=np.float32)
            decision = s_r[np.argmax(self.model(possible_states, training=False).numpy())]
            self.game.GameLoopCNN(*decision, gui = True)





model = TetrisModel(1e-4, 100, 0.999)
model.load()
for i in range(1000):
    model.optimize(batch_size=64)
    model.play(4)
    model.info()
model.save()
model.plot()
model.play_test(1000)


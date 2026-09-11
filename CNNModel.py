import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import random

class TetrisModel: # pooling olmamasi sart cunku indirgeme yapoiyor pooling. daha cok image icin feature extractionda parametreleri azaltmak icin kullaniliyor
    def __init__(self):
        #PARALLEL CONVOLUTION FILTERS
        inputs = layers.Input(shape=(20, 10, 1)) # yukseklik genislik kanal
        branch1x3 = layers.Conv2D(32, (1, 3), activation='relu', padding='same')(inputs) #yatay bilgiler
        branch5x1 = layers.Conv2D(32, (5, 1), activation='relu', padding='same')(inputs) #dikey bilgiler
        branch3x3 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs) #genjel feature extraction
        x = layers.Concatenate(axis=-1)([branch1x3, branch5x1, branch3x3])
        x = layers.Conv2D(16, (1, 1), activation='relu')(x) #parametre sayisini azaltmak icin
        x = layers.Flatten()(x)
        x = layers.Dense(64, activation='relu')(x)
        output = layers.Dense(1, activation='linear')(x) # negatif cezalari isleyebilmesi icin linear

        self.model = models.Model(inputs=inputs, outputs=output, name='TetrisModel')
        self.target_model = models.Model(inputs=inputs, outputs=output, name='TargetTetrisModel')
        self.model.summary()

        self.memory = [] # 20x10np.array (s), int (reward), int (next_piece_enum), done

    def sample(self, sample_size):
        return random.sample(self.memory, sample_size)

    def optimize(self, batch_size):
        if len(self.memory) < batch_size:
            return
        batch = self.sample(batch_size)
        states = np.array([np.expand_dims(item[0], axis=-1) for item in batch], dtype=np.float32) #(20, 10, 1)
        rewards = np.array([item[1] for item in batch], dtype=np.float32)
        max_future_qs = np.zeros(batch_size)
        future_states = []
        fstates_id = [] # future statelerin sirasini kaybetmemek icin
        filter_id = [] # kendi aralarinda hangi gruba ait olduklarini unutmamak icin
        id = 0
        for i in range(batch_size):
            if (not batch[i][3]):
                for s in game.getStates(batch[i][0], batch[i][2]): # game.getStates varmis gibi yazdim ama yok su an...
                    future_states.append(s)
                    filter_id.append(id)
                fstates_id.append(i)
                id += 1

        if id > 0: #hic gelen bir sey yoksa done hepsi icin true ise bosuna gpu calismasin
            future_states = np.expand_dims(np.array(future_states, dtype=np.float32),
                                           axis=-1)  # (batch_size, 20, 10 ,1)
            q_preds = self.target_model(future_states, training=False)
            q_preds = tf.squeeze(q_preds, axis=-1)
            segment_ids = tf.constant(filter_id, dtype=tf.int32)
            max_future_qs[fstates_id] = tf.math.segment_max(q_preds, segment_ids).numpy()
        target = rewards + 0.99 * max_future_qs




TetrisModel()
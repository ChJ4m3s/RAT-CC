from tensorflow.keras import Model
from tensorflow.keras.layers import LSTM, Input, Dense, TimeDistributed, Dropout, Lambda, Bidirectional
from tensorflow.keras.optimizers import Adam
from lib import *


def rat_cc(shape: tuple, emb_size: int, n_layers, n_hidden, l):
    input_layer = Input(shape=(None, shape[1]), name='input')
    embedding = Bidirectional(LSTM(n_hidden, return_sequences=True))(input_layer)
    embedding = Dropout(0.2)(embedding)

    for _ in range(n_layers):
        embedding = Bidirectional(LSTM(n_hidden, return_sequences=True))(embedding)
        embedding = Dropout(0.2)(embedding)

    embedding = Bidirectional(LSTM(emb_size, return_sequences=False), name="embedding")(embedding)

    repeat_layer = Lambda(repeat)([input_layer, embedding])

    decoder = Bidirectional(LSTM(n_hidden, return_sequences=True))(repeat_layer)
    decoder = Dropout(0.2)(decoder)

    for _ in range(n_layers):
        decoder = Bidirectional(LSTM(n_hidden, return_sequences=True))(decoder)
        decoder = Dropout(0.2)(decoder)

    reconstruction_output = TimeDistributed(Dense(shape[1]), name="reconstruction")(decoder)

    model = Model(inputs=[input_layer], outputs=[reconstruction_output, embedding])

    model.compile(
        loss={
            'reconstruction': mean_absolute_percentage_error,
            'embedding': cosine_similarity_loss
        },
        loss_weights={
            'reconstruction': 2 - l,
            'embedding': 1 + l
        },
        optimizer=Adam(learning_rate=1e-4),
        metrics={
            'reconstruction': 'mse',
        }
    )

    return model
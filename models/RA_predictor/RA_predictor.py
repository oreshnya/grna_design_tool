from keras.models import Sequential
from keras.layers import Input, Conv2D, MaxPool2D, Dropout, Flatten, Dense
from keras.optimizers import Adam

def build_relative_activity_predictor(weights_path=None):
    
    model = Sequential()
    model.add(Input(shape=(7, 26, 1), name='input'))

    model.add(Conv2D(filters=32, kernel_size=(4,4), activation='relu', padding='same', 
                        data_format='channels_last', name='conv_1'))
    model.add(Conv2D(filters=32, kernel_size=(4,4), activation='relu', padding='same', 
                        data_format='channels_last', name='dense_2'))
    model.add(MaxPool2D(pool_size=(1,2), padding='same', data_format='channels_last', name='dense_3'))
    model.add(Dropout(0.25,  name="dropout_1"))
    model.add(Flatten(name="flatten_1"))
    model.add(Dense(units=256, activation='sigmoid', name='dense_4'))
    model.add(Dropout(0.25, name="dropout_2"))
    model.add(Dense(units=128, activation='sigmoid', name='float_dense_5'))
    model.add(Dropout(0.25, name="dropout_3"))
    model.add(Dense(1, activation='linear', name='float_dense_7'))

    model.compile(
        loss='log_cosh',
        optimizer='adam',
        metrics=['mean_absolute_error']
    )

    if weights_path:
        model.load_weights(weights_path)
        print("✅ Weights loaded")

    model.summary()
    return model
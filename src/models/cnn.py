"""CNN de calidad de frutas. Responsable: Andrés Fajardo."""


def build_cnn(input_shape=(128, 128, 3)):
    from tensorflow import keras
    from tensorflow.keras import layers

    # Los filtros aprenden desde cero; no se utilizan pesos preentrenados.
    model = keras.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv2D(32, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(2),
        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(2),
        layers.Conv2D(128, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(2),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.4),
        layers.Dense(3, activation='softmax'),
    ], name='cnn_frutas')
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),
                  loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

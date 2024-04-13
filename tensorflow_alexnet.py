# Tensorflow Alexnet model implementation
from tensorflow.keras import datasets, layers, models

# Alexnet
kernel_regular = None
n_classes = 1000

model = models.Sequential()
model.add(
    layers.Conv2D(filters=96, kernel_size=(11, 11), strides=(4, 4), padding="valid")
)
model.add(layers.Activation("relu"))
model.add(layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))

model.add(
    layers.Conv2D(
        filters=256,
        kernel_size=(5, 5),
        strides=(1, 1),
        padding="same",
        kernel_regularizer=kernel_regular,
    )
)
model.add(layers.Activation("relu"))
model.add(layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))

model.add(
    layers.Conv2D(
        filters=384,
        kernel_size=(3, 3),
        strides=(1, 1),
        padding="same",
        kernel_regularizer=kernel_regular,
    )
)
model.add(layers.Activation("relu"))

model.add(
    layers.Conv2D(
        filters=384,
        kernel_size=(3, 3),
        strides=(1, 1),
        padding="same",
        kernel_regularizer=kernel_regular,
    )
)
model.add(layers.Activation("relu"))

model.add(
    layers.Conv2D(
        filters=256,
        kernel_size=(3, 3),
        strides=(1, 1),
        padding="same",
        kernel_regularizer=kernel_regular,
    )
)
model.add(layers.Activation("relu"))
model.add(layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))

model.add(layers.Flatten())
model.add(layers.Dense(units=4096, activation="relu"))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(units=4096, activation="relu"))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(units=n_classes, activation="softmax"))

model.summary()

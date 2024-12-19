import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, ReLU
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.metrics import SparseCategoricalAccuracy
from tensorflow.keras.regularizers import l2

import numpy as np
import mnist_loader

# Data Load
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
training_data = list(training_data)
validation_data = list(validation_data)
test_data = list(test_data)

x_train = np.vstack([x.flatten() for x, _ in training_data]).astype("float32")
y_train = np.array([np.argmax(y) if isinstance(y, np.ndarray) else y for _, y in training_data], dtype="int32")

x_val = np.vstack([x.flatten() for x, _ in validation_data]).astype("float32")
y_val = np.array([np.argmax(y) if isinstance(y, np.ndarray) else y for _, y in validation_data], dtype="int32")

x_test = np.vstack([x.flatten() for x, _ in test_data]).astype("float32")
y_test = np.array([np.argmax(y) if isinstance(y, np.ndarray) else y for _, y in test_data], dtype="int32")

print(f"x_train shape: {x_train.shape}")
print(f"y_train shape: {y_train.shape}")

# Create model function
def create_model(input_size, hidden_sizes, output_size):
    inputs = tf.keras.Input(shape=(input_size,))
    x = Dense(hidden_sizes[0], activation='relu', kernel_regularizer=l2(0.001))(inputs)

    for hidden_size in hidden_sizes[1:]:
        x = Dense(hidden_size, activation='relu', kernel_regularizer=l2(0.001))(x)

    outputs = Dense(output_size, activation='softmax')(x)
    return tf.keras.Model(inputs=inputs, outputs=outputs)

# Training function
def train_model(model, x_train, y_train, x_val, y_val, learning_rate, epochs, batch_size):
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=0.1, decay_steps=1000, decay_rate=0.96)
    optimizer = SGD(learning_rate=lr_schedule)

    loss_fn = SparseCategoricalCrossentropy()
    model.compile(optimizer=optimizer, loss=loss_fn, metrics=[SparseCategoricalAccuracy()])

    early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', patience=10, restore_best_weights=True)

    history = model.fit(
    x_train, y_train,
    validation_data=(x_val, y_val),
    epochs=epochs,
    batch_size=batch_size,
    verbose=2,
    callbacks=[early_stopping])

    return history

# Evaluation model
def evaluate_model(model, x_test, y_test):
    _, accuracy = model.evaluate(x_test, y_test, verbose=0)
    print(f"Test accuracy: {accuracy:.4f}")
    return accuracy

# Parameters
input_size = x_train.shape[1]
output_size = 10
layer_configs = [[128], [128, 64], [64, 64, 64]]
learning_rates = [0.1, 0.5]
batch_size = 64
epochs = 150

results = []
for hidden_sizes in layer_configs:
    for learning_rate in learning_rates:
        print(f"\nRunning experiment with layers: {hidden_sizes}, learning rate: {learning_rate}")
        model = create_model(input_size, hidden_sizes, output_size)

        train_model(model, x_train, y_train, x_val, y_val,
                    learning_rate=learning_rate, epochs=epochs, batch_size=batch_size)

        accuracy = evaluate_model(model, x_test, y_test)

        results.append({"layers": hidden_sizes, "learning_rate": learning_rate, "test_accuracy": accuracy})

for result in results:
    print(result)

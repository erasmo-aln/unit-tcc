import tensorflow as tf

from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras import Model

# General pipeline of an end-to-end project:

# 1) Get the dataset:
mnist = tf.keras.datasets.mnist 
(x_train, y_train), (x_test, y_test) = mnist.load_data()
train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(10000).batch(32)
test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)

# 2) Define the model:
class MyModel(Model):
    def __init__(self): # Define the layers
        super(MyModel, self).__init__()
        self.conv1 = Conv2D(32, 3, activation='relu')
        self.flatten = Flatten()
        self.d1 = Dense(128, activation='relu')
        self.d2 = Dense(10)

    def call(self, x): # Define the operations
        x = self.conv1(x)
        x = self.flatten(x)
        x = self.d1(x)
        return self.d2(x)

# 3) Create an instance of the model
model = MyModel()

# 4) Define loss and activation functions:
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
optimizer = tf.keras.optimizers.Adam()

# 5) Define metrics:
train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

validation_loss = tf.keras.metrics.Mean(name='test_loss')
validation_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

# 6) Define the training and validations steps with subclassing:
def step(images, labels):
    with tf.GradientTape() as tape:
        predictions = model(images, training=True) # The output of model
        loss = loss_object(labels, predictions) # The loss with predictions
    gradients = tape.gradient(loss, model.trainable_variables) # Get the gradients
    optimizer.apply_gradients(zip(gradients, model.trainable_variables)) # Apply the gradient descent with the optimizer
    train_loss(loss) # Get losses over epochs
    train_accuracy(labels, predictions) # Get accuracies over epochs

def validation_step(images, labels): # Do the same thing as above, but for validate
    predictions = model(images, training=False)
    t_loss = loss_object(labels, predictions)
    validation_loss(t_loss)
    validation_accuracy(labels, predictions)

# 7) Train and evaluate the model:
EPOCHS = 5

for epoch in range(EPOCHS):
    # Reset the metrics at the start of the next epoch
    train_loss.reset_states()
    train_accuracy.reset_states()
    validation_loss.reset_states()
    validation_accuracy.reset_states()

    for images, labels in train_dataset:
        step(images, labels)

    for test_images, test_labels in test_dataset:
        validation_step(test_images, test_labels)

    # Print results:
    template = 'Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}'
    print(template.format(epoch + 1,
                        train_loss.result(),
                        train_accuracy.result() * 100,
                        validation_loss.result(),
                        validation_accuracy.result() * 100))

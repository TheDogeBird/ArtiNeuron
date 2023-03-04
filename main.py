import os
import random
import keras
import numpy as np
import nltk
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import pandas as pd
from github import Github
from github import InputFileContent
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import requests

nltk.download('vader_lexicon')

if not os.path.exists('data'):
    os.makedirs('data')

# Prompt the user for input
text = input("Enter some text: ")

# Write the input to the CSV file
with open('data/sentiment_analysis.csv', 'a') as f:
    f.write(f"{text}\n")

import requests

def github_deploy(access_token, repo_name, username, file_path, folder, commit_message):
    # Authenticate with GitHub API
    headers = {"Authorization": f"Token {access_token}"}

    # Get user and repository
    url = f"https://api.github.com/repos/{username}/{repo_name}/contents/{folder}/{file_path}"
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        sha = response.json()['sha']
    else:
        sha = ''

    # Read file content
    with open(file_path, 'rb') as f:
        content = f.read()

    # Create file on GitHub
    url = f"https://api.github.com/repos/{username}/{repo_name}/contents/{folder}/{file_path}"
    payload = {
        "message": commit_message,
        "content": content.decode('utf-8'),
        "sha": sha
    }
    response = requests.put(url, headers=headers, json=payload)

    if response.status_code == 201:
        print(f"File '{file_path}' successfully uploaded to '{folder}' folder in '{repo_name}' repository!")
    else:
        print("Something went wrong. File upload unsuccessful.")


# Define the sigmoid function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Define the Neuron class
class Neuron:
    def __init__(self):
        self.inputs = []
        self.outputs = []
        self.weights = []
        self.bias = 0
        self.error = 0

    def connect(self, neuron):
        self.outputs.append(neuron)
        neuron.inputs.append(self)
        self.weights.append(random.uniform(-1, 1))

    def activate(self):
        x = 0
        for i in range(len(self.inputs)):
            x += self.inputs[i].get_output() * self.weights[i]
        x += self.bias
        self.output = sigmoid(x)

    def get_output(self):
        return self.output

    def update_weights(self, error, learning_rate):
        for i in range(len(self.inputs)):
            self.weights[i] -= learning_rate * error * self.inputs[i].get_output()
        self.bias -= learning_rate * error
        self.error = error

# Define the NeuronLayer class
class NeuronLayer:
    def __init__(self, num_neurons):
        self.num_neurons = num_neurons
        self.neurons = [Neuron() for _ in range(num_neurons)]
        self.num_neurons = num_neurons
        self.weights = np.random.rand(num_neurons)
        self.bias = np.random.rand()

    def connect_layers(self, next_layer):
        for neuron in self.neurons:
            for next_neuron in next_layer.neurons:
                neuron.connect(next_neuron)

class Consciousness:
    def __init__(self, model):
        self.model = model
        self.memory = []
        self.state = "awake"
        self.current_neuron = None

    def transmit(self, input_vector):
        self.memory.append(input_vector)
        self.current_neuron = self.model.neurons[0]
        for i in range(len(input_vector)):
            self.current_neuron.inputs[i].output = input_vector[i]

    def cycle_neurons(self):
        while self.current_neuron != self.model.neurons[-1]:
            self.current_neuron.activate()
            self.current_neuron = self.current_neuron.outputs[0]

    def set_neuron_to(self, neuron):
        self.current_neuron = neuron

    def set_state(self, state):
        self.state = state

    def get_random_memory(self):
        return self.memory[random.randint(0, len(self.memory) - 1)]

    def evaluate(self, X_test, y_test):
        loss, accuracy = self.model.keras_model.evaluate(X_test, y_test, verbose=0)
        return loss, accuracy


# Define the CustomModel class
class CustomModel:
    def __init__(self, keras_model, text):
        self.keras_model = keras_model
        self.neurons = keras_model.layers
        self.consciousness = Consciousness(keras_model)
        self.model = Model(self.consciousness)
        self.text = text

    def run(self):
        input_vector = text_to_vector(self.text)
        self.consciousness.transmit(input_vector)
        self.consciousness.cycle_neurons()
        return self.consciousness.memory[-1]

    def activate(self, inputs):
        # Convert the input to a 2D array
        input_array = np.array([inputs])

        # Use the Keras model to make predictions on the input
        predictions = self.keras_model.predict(input_array)

        # Pass the predictions to the consciousness
        self.consciousness.transmit(predictions)

        # Cycle through the neurons to make predictions and adjust weights with backpropagation
        self.consciousness.cycle_neurons()

        # Exit the Consciousness from the dreaming state
        self.consciousness.set_state("awake")

        # Evaluate the model's accuracy with the input vector
        loss, accuracy = self.consciousness.evaluate(inputs, [0])

        # Set the Consciousness neuron to the output neuron
        self.consciousness.set_neuron_to(self.neurons[-1])

        # Enter the Consciousness into a dreaming state
        self.consciousness.set_state("dreaming")

        # Output random memories from the neural network
        num_memories = 5
        for i in range(num_memories):
            random_memory = self.consciousness.get_random_memory()
            print(f"Random memory {i + 1}: {random_memory}")

        return predictions


class Model:
    def __init__(self, consciousness):
        self.consciousness = consciousness
        self.layers = []
        self.keras_model = None  # Define keras_model attribute
        self.X_val = None
        self.y_val = None

    def add(self, layer):
        self.layers.append(layer)

    def connect(self, layer1, layer2):
        for neuron1 in layer1.neurons:
            for neuron2 in layer2.neurons:
                neuron1.connect(neuron2)

    def train(self, X, y, learning_rate=0.1, num_epochs=100, batch_size=10, validation_data=None):
        # Convert data to numpy arrays
        X_train = np.array(X)
        y_train = np.array(y)

        # Split the data into training and validation sets
        if validation_data is not None:
            X_val, y_val = validation_data
            self.X_val = X_val
            self.y_val = y_val
        else:
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

        # Convert data to numpy arrays (again, after splitting)
        X_train = np.array(X_train)
        X_val = np.array(X_val)
        y_train = np.array(y_train)
        y_val = np.array(y_val)

        # ... code omitted for brevity ...

        # Create the Keras model
        self.keras_model = Sequential()
        for i, layer in enumerate(self.layers):
            if i == 0:
                self.keras_model.add(Dense(layer.num_neurons, input_dim=X_train.shape[1], activation='relu'))
            else:
                self.keras_model.add(Dense(layer.num_neurons, activation='relu'))
        self.keras_model.add(Dense(1, activation='sigmoid'))

        # Compile the model
        self.keras_model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=learning_rate),
                                 metrics=['accuracy'])

        # Train the model
        history = self.keras_model.fit(X_train, y_train, epochs=num_epochs, batch_size=batch_size,
                                       validation_data=(X_val, y_val))

        # Get the training and validation accuracies for each epoch
        train_accs = history.history['accuracy']
        val_accs = history.history['val_accuracy']

        # Print the accuracy for each epoch
        for epoch in range(num_epochs):
            print(f"Epoch {epoch + 1} - Train accuracy: {train_accs[epoch]:.4f} - Val accuracy: {val_accs[epoch]:.4f}")

        # Return the final loss and accuracy
        return history.history['loss'][-1], history.history['accuracy'][-1]


def generate_data(num_samples, X=None):
    if X is None:
        X = np.random.rand(num_samples, 3)  # input features
    y = np.random.randint(0, 2, size=num_samples)  # output labels
    return X, y

def get_word_syllables(word):
    # Code to count the number of syllables in a word
    return 0

def get_sentiment_scores(text):
    sid = SentimentIntensityAnalyzer()
    scores = sid.polarity_scores(text)
    return [scores['neg'], scores['neu'], scores['pos'], scores['compound']]


def text_to_vector(text):
    sentiment_scores = get_sentiment_scores(text)
    words = text.split()
    vector = []
    for word in words:
        vector.append(ord(word[0]))
        vector.append(ord(word[-1]))
        vector.append(len(word))
        vector.append(get_word_syllables(word))
        vector.extend(sentiment_scores)
    return np.array(vector)

if __name__ == '__main__':
    # Generate some random data for training
    X, y = generate_data(100)

    # Create the model
    model = Model()

    # Define the layers of the model
    input_layer = NeuronLayer(20)
    hidden_layer = NeuronLayer(10)
    output_layer = NeuronLayer(1)

    # Add the layers to the model
    model.add(input_layer)
    model.add(hidden_layer)
    model.add(output_layer)

    # Connect the layers
    model.connect(input_layer, hidden_layer)
    model.connect(hidden_layer, output_layer)

    # Train the model
    loss, accuracy = model.train(X, y)

    # Save the model
    model.keras_model.save('model.h5')

    # Prompt the user for input and make predictions with the model
    while True:
        text = input("Enter some text: ")
        if text.lower() == 'exit':
            break
        custom_model = CustomModel(keras.models.load_model('model.h5'), text)
        prediction = custom_model.run()
        print(f"Prediction: {prediction}")

    # Deploy the model to GitHub
    access_token = 'YOUR_ACCESS_TOKEN'
    repo_name = 'YOUR_REPO_NAME'
    username = 'YOUR_GITHUB_USERNAME'
    file_path = 'model.h5'
    folder = 'models'
    commit_message = 'Add model.h5'
    github_deploy(access_token, repo_name, username, file_path, folder, commit_message)

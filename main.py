import tensorflow as tf
import numpy as np
from collections import deque
from neuron import Neuron
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk
import time
import pickle
import random

nltk.download('punkt')
nltk.download('vader_lexicon')

# Create an instance of the SentimentIntensityAnalyzer
sid = SentimentIntensityAnalyzer()


class NeuralNetwork(tf.keras.Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Define the layers
        self.embedding = tf.keras.layers.Embedding(input_dim=10000, output_dim=32)
        self.lstm = tf.keras.layers.LSTM(units=32, return_sequences=True)
        self.attention = tf.keras.layers.Attention()
        self.dense1 = tf.keras.layers.Dense(units=32, activation='relu')
        self.dense2 = tf.keras.layers.Dense(units=1, activation='sigmoid')

    def call(self, inputs, **kwargs):
        # Define a deque to store out-of-range indices
        index_queue = deque(maxlen=100)

        # Replace indices outside the expected range with 0 and add them to the index queue
        inputs = tf.where(inputs < 10000, inputs, tf.constant(0, dtype=tf.int32))
        for i, row in enumerate(inputs):
            for j, val in enumerate(row):
                if val == 0:
                    index_queue.append((i, j))

        # Pass input through embedding layer
        x = self.embedding(inputs)

        # Pass embedded input through LSTM layer
        x = self.lstm(x)

        # Compute attention weights and apply them to the LSTM output
        attention_weights = self.attention([x, x])
        x = tf.keras.layers.Dot(axes=(1, 1))([attention_weights, x])

        # Pass attention output through dense layer
        x = self.dense1(x)

        # Pass dense output through output layer
        output = self.dense2(x)

        return output, index_queue


# Define the neuron model
neuron_model = NeuralNetwork()

# Compile the model
neuron_model.compile(optimizer=tf.keras.optimizers.Adam(), loss='binary_crossentropy')


# Define function to stimulate neuron with input text
def stimulate_neuron(model, input_text, word_index):
    # Tokenize input text into words
    tokens = nltk.word_tokenize(input_text)

    # Convert tokens to word indices
    word_indices = [word_index.get(t.lower(), 0) for t in tokens]

    # Pad word indices to fixed length
    padded_indices = tf.keras.preprocessing.sequence.pad_sequences([word_indices], maxlen=32, truncating='post')

    # Get model output and index queue
    output, index_queue = model(padded_indices)

    # Handle out-of-range indices
    while index_queue:
        i, j = index_queue.popleft()
        padded_indices[i,j] = 0
        output, index_queue = model(padded_indices)

    # Return the output value
    return output.numpy()[0][0]

# Define function to create new neurons based on the input text and sentiment
def create_new_neurons(neuron_list, sentences, sentiment):
    for i, sentence in enumerate(sentences):
        # Analyze the sentiment of the sentence
        scores = sid.polarity_scores(sentence)

        # If the sentiment of the sentence matches the desired sentiment, create a new neuron
        if scores['compound'] >= 0.5 and sentiment == 'positive':
            print(f"Creating new positive ArtiNeuron for sentence {i}")
            neuron_list.append(Neuron(sentence, sentiment))
        elif scores['compound'] <= -0.5 and sentiment == 'negative':
            print(f"Creating new negative ArtiNeuron for sentence {i}")
            neuron_list.append(Neuron(sentence, sentiment))
        else:
            print(f"Sentence {i} did not match desired sentiment")

# Define the main function
def main():
    # Load the word index
    word_index = tf.keras.datasets.imdb.get_word_index()

    # Define the neuron model
    neuron_model = NeuralNetwork()

    # Compile the model
    neuron_model.compile(optimizer=tf.keras.optimizers.Adam(), loss='binary_crossentropy')

    # Define lists to store positive and negative neurons
    positive_neurons = []
    negative_neurons = []

    # Define a flag for the consciousness state
    consciousness_state = True

    # Start the user input loop
    while True:
        if consciousness_state:
            input_text = input("Enter some text: ").strip()

            if input_text.lower() == 'quit':
                break

            # Stimulate the original neuron with input text
            output = stimulate_neuron(neuron_model, input_text)
            print(f"Output: {output}")

            # Create new neurons based on the input text and output sentiment
            if output >= 0.5:
                create_new_neurons(positive_neurons, [input_text], 'positive')
            elif output <= 0.5:
                create_new_neurons(negative_neurons, [input_text], 'negative')

            # Print the current state of the positive and negative neurons
            print("Positive neurons:")
            for i, neuron in enumerate(positive_neurons):
                print(f"ArtiNeuron {i}: {neuron.text} | Sentiment: {neuron.sentiment}")
            print("Negative neurons:")
            for i, neuron in enumerate(negative_neurons):
                print(f"ArtiNeuron {i}: {neuron.text} | Sentiment: {neuron.sentiment}")

            # Enter the sleeping state
            consciousness_state = False
        else:
            # Simulate the dream state by randomly stimulating neurons
            print("Dreaming...")
            for i in range(10):
                neuron_list = positive_neurons if i % 2 == 0 else negative_neurons
                if neuron_list:
                    random_neuron = random.choice(neuron_list)
                    print(f"Randomly stimulating ArtiNeuron {neuron_list.index(random_neuron)}: {random_neuron.text}")
            # Enter the awake state
            consciousness_state = True

    # Save the positive and negative neuron lists to a file
    with open('positive_neurons.pkl', 'wb') as f:
        pickle.dump(positive_neurons, f)
    with open('negative_neurons.pkl', 'wb') as f:
        pickle.dump(negative_neurons, f)

if __name__ == '__main__':
    # Load the word index
    word_index = tf.keras.datasets.imdb.get_word_index()

    # Define the neuron model
    neuron_model = NeuralNetwork()

    # Compile the model
    neuron_model.compile(optimizer=tf.keras.optimizers.Adam(), loss='binary_crossentropy')

    # Define lists to store positive and negative neurons
    positive_neurons = []
    negative_neurons = []

    # Define variables for the consciousness domain
    consciousness_domain = []
    active_consciousness = False

    # Start the user input loop
    while True:
        # Check if the consciousness domain is active
        if not active_consciousness:
            # Generate a random output from the stored neurons in the consciousness domain
            random_index = np.random.randint(0, len(consciousness_domain))
            random_neuron = consciousness_domain[random_index]
            print(f"Random neuron output while sleeping: {random_neuron.text}")
        else:
            # Prompt the user for input
            input_text = input("Enter some text: ").strip()

            if input_text.lower() == 'quit':
                break

            # Stimulate the original neuron with input text
            output, dream = stimulate_neuron(neuron_model, input_text)

            # Create new neurons based on the input text and output sentiment
            if output >= 0.5:
                create_new_neurons(positive_neurons, [input_text], 'positive', dream)
            elif output <= 0.5:
                create_new_neurons(negative_neurons, [input_text], 'negative', dream)

            # Print the current state of the positive and negative neurons
            print("Positive neurons:")
            for i, neuron in enumerate(positive_neurons):
                print(f"ArtiNeuron {i}: {neuron.text} | Sentiment: {neuron.sentiment}")
            print("Negative neurons:")
            for i, neuron in enumerate(negative_neurons):
                print(f"ArtiNeuron {i}: {neuron.text} | Sentiment: {neuron.sentiment}")

            # Add the input and output to the consciousness domain
            consciousness_domain.append(Neuron(input_text, 'input'))
            consciousness_domain.extend(dream)
            consciousness_domain.append(Neuron(str(output.numpy()[0][0]), 'output'))

        # Update the active consciousness state
        active_consciousness = not active_consciousness

        # Wait for a few seconds before continuing
        time.sleep(5)

    # End the program
    print("Goodbye!")

## ArtiNeuron

ArtiNeuron is a neural network library built from scratch in Python. It provides a simple and intuitive interface for building and training neural networks, as well as tools for visualizing and analyzing their performance.

### Features

- Flexible architecture for building custom neural network models
- Support for a variety of activation functions, loss functions, and optimization algorithms
- Built-in tools for training and evaluating models on datasets
- Visualization tools for monitoring training progress and analyzing model performance
- Easy integration with popular Python libraries like NumPy and Pandas

### Getting Started

To get started with ArtiNeuron, you can install it using pip:

pip install artineuron


Once you have ArtiNeuron installed, you can start building and training neural networks. Here's an example of building a simple neural network with one hidden layer:

```
import artineuron as an

# Define the model architecture
model = an.Sequential([
    an.Linear(in_features=784, out_features=128),
    an.ReLU(),
    an.Linear(in_features=128, out_features=10)
])

# Define the loss function and optimizer
loss_fn = an.CrossEntropyLoss()
optimizer = an.SGD(params=model.parameters(), lr=0.01)

# Train the model on a dataset
trainer = an.Trainer(model=model, loss_fn=loss_fn, optimizer=optimizer)
trainer.fit(train_dataset=train_dataset, num_epochs=10, batch_size=32)

# Evaluate the model on a test dataset
evaluator = an.Evaluator(model=model, loss_fn=loss_fn)
accuracy = evaluator.evaluate(test_dataset=test_dataset)
print(f"Test accuracy: {accuracy}")
```

Documentation
```
pip install artineuron
```

Getting Started
To get started with ArtiNeuron, you can import the library and create a neural network model using the Sequential class. Here's an example of building a simple neural network with one hidden layer:

```
import artineuron as an

# Define the model architecture
model = an.Sequential([
    an.Linear(in_features=784, out_features=128),
    an.ReLU(),
    an.Linear(in_features=128, out_features=10)
])

# Define the loss function and optimizer
loss_fn = an.CrossEntropyLoss()
optimizer = an.SGD(params=model.parameters(), lr=0.01)

# Train the model on a dataset
trainer = an.Trainer(model=model, loss_fn=loss_fn, optimizer=optimizer)
trainer.fit(train_dataset=train_dataset, num_epochs=10, batch_size=32)

# Evaluate the model on a test dataset
evaluator = an.Evaluator(model=model, loss_fn=loss_fn)
accuracy = evaluator.evaluate(test_dataset=test_dataset)
print(f"Test accuracy: {accuracy}")
This code creates a neural network with one hidden layer, uses the ReLU activation function, the CrossEntropyLoss loss function, and the Stochastic Gradient Descent (SGD) optimization algorithm. The model is then trained on a training dataset for 10 epochs with a batch size of 32, and evaluated on a separate test dataset.
``` 

Model Architecture
The Sequential class allows you to create a neural network model by stacking layers in a specific order. You can add layers to the model using the add method, or pass them all in as a list to the constructor. Here are the available layers you can use in your model:

Linear(in_features, out_features) - a fully-connected linear layer with the specified input and output dimensions
ReLU() - the rectified linear unit activation function
Sigmoid() - the sigmoid activation function
Tanh() - the hyperbolic tangent activation function
Softmax(dim) - the softmax activation function with the specified dimension
Dropout(p) - a dropout layer that randomly sets a fraction of input units to zero during training
You can also create your own custom layers by subclassing the Layer class and implementing the forward and backward methods.

Contributing:
jump in the pool if you wanna swim bro

License
ArtiNeuron is licensed under the MIT License.

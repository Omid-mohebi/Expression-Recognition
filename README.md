# Facial Expression Recognition Model

This project implements a neural network to classify facial expressions using data from the FER2013 dataset.

## Prerequisites
- Python 3.x
- Libraries: `numpy`, `pandas`

## Dataset
- Download the dataset from [Kaggle](https://www.kaggle.com/datasets/nicolejyt/facialexpressionrecognition).

## Code Overview

1. **Data Preprocessing**
   - Loads and normalizes the dataset.
   - Balances classes by oversampling.
   - Converts labels to one-hot encoding.
   - Splits data into 70% training and 30% testing sets.

2. **Model Functions**
   - Implements activation functions (`sigmoid`, `softmax`, `relu`).
   - Defines forward propagation for two-layer neural network.

3. **Training Loop**
   - Trains the model using backpropagation with regularization.

## Hyperparameters
- `learning_rate`: 0.001
- `epochs`: 1000
- `reg`: 0.01

## Results

With the current hyperparameters it reached 58% accuracy, it can be improved by adding more layers and playing with other hyperparameters.

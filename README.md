# Table of Contents
- Neural Networks
- Backpropagation
- Activation Functions
- Argmax and Softmax
- Cross Entropy Loss
- CNN
- RNN
- LSTM
- Word Embedding
- Attention
- Transformer
- Project Guidelines

## Neural Networks
Neural networks are computer models modelled like the human brain that consist of nodes (like neurons) that are interconnected in layers and used for finding patterns, approximating functions, and learning from data.

## Backpropagation
Backpropagation is the dominant algorithm for training a neural network and determines the gradient of the loss function with respect to each parameter and updates the parameters using gradient descent to minimize error.

## Activation Functions
- Sigmoid function: Outputs between 0 and 1; great for probabilistic models.
- Softplus: smooth, differentiable version of ReLU; maps input-neurons to positive values.
- ReLU (Rectified Linear Unit): outputs the input, if input is positive, otherwise 0, used extensively because of simplicity.

## Argmax and Softmax
- Argmax: Returns the index of the largest value in a vector, often used to pick the most probable class.
- Softmax: Converts a vector of scores into probabilities that sum to 1, useful for multi-class classification.

## Cross Entropy Loss
Cross entropy measures the differences between predicted probability distribution and true distribution; loss function commonly used for classification.

## Convolutional Neural Networks (CNN)
CNN's structure is particularly beneficial for data with a grid-like topology as seen in images. CNN's use of convolutional layers enables them to automatically and adaptively learn spatial hierarchies of features.

## Recurrent Neural Networks (RNN)
RNNs are specifically tailored towards sequential data, exhibiting a hidden state where the hidden state of each step encompasses information from the previous steps and enables learning temporal patterns.

## Long Short-Term Memory (LSTM)
Handling an RNNs inherent weight updates from the previous steps, LSTMs (Long Short Term Memories) are a specific type of RNN that can learn long-term dependencies through gating mechanisms to effectively mitigate the vanishing gradient problem found in standard RNNs.

## Word Embedding
Word embedding entails packing words into a dense vector space, where similar vectors are assigned to semantically similar words. This representation increases the neural network’s ability to learn language.

## Attention
Attention mechanisms show that the model can pay attention to specific parts of the input sequence, varying weights for different steps, allowing the model to improve their performance on tasks such as translation and summarization.

## Transformer
Instead of permitting recurrence through the hidden state, a more effective structure known as a transformer, replace recurrence with stacked self-attention and feed-forward layers to allow highly parallelized training, enabling the model to perform particularly well on tasks with a language situation as well as sequence transduction tasks.

## How the Final Code Works
This project implements a Transformer-based language model at the character level, trained on a text dataset. Here’s how the main components come together:

- **Data Preparation:**  
  The code will download a text file, encode the text as a sequence of unique character indices, and split it into training and validation datasets. There are utilities to easily sample batches for training.

- **Network Architecture:**  
  The model is a small GPT-like Transformer model, made of several stacked blocks where each block has:  
  - Multi-head causal self-attention (enabling each character to use the context of the previous characters in the sequence)  
  - Feed-forward networks (to process features at each position)  
  - Layer normalization and residual connections (to allow for efficient training)

- **Training Loop:**  
  During training, random sections of the text, sampled from a batch, are given to the model. For every position in the text, the model will predict the character, and the loss is reported in intervals. The optimizer will adjust weights to lower that loss.

- **Text Generation:**  
  After training, the model can produce new text by predicting one character at a time, by using an initial context. At each step, it samples the next character from the probability distribution output by the model.

# Parallel LSTM training for sequence prediction from sequential data

Vietnam National University - University of Science - Falculty of Information Technology

CSC14116 - Applied Parallel Programming

19120454 - Bui Quang Bao

# Introduction

In this project, I will analyze and parallel the LSTM model (a RNN - Recurrent Neural Network) in order to improve its training speed and efficiency. By utilizing parallel processing and GPU computing, the model will be able to handle larger datasets and have shorter training duration. The specific task that I want to apply using the LSTM model in this project is time-series prediction - sequence prediction from sequential data. I will implement a raw LSTM model using only Numpy library, analyze, parallelize using Numba library, and measure the efficiency of the parallel version over the sequential version.

Keywords: `Parallel Processing`, `GPU Computing`, `NVIDIA CUDA`, `Recurrent Neural Network (RNN)`, `Long Short-Term Memory (LSTM)`

# Background

## About RNN and LSTM

A recurrent neural network (RNN) trains on input containing sequences of data as it discovers the relationships between various parts of the input that are time-dependent. An RNN, for instance, can learn about the relationships between words if we give it a sentence's worth of words as input. In doing so, it can also learn about the rules of grammar, such as the connections between verbs and adverbs, etc.

LSTM stands for Long Short-Term Memory. It is a type of recurrent neural network (RNN) that is designed to handle the problem of vanishing gradients in traditional RNNs. LSTM networks have the ability to selectively remember or forget information over long periods of time, making them particularly effective for tasks that involve sequential data, such as speech recognition, language translation, and time series prediction.

## LSTM Architecture

A basic LSTM neural network contains an input layer, hidden layer and output layer, containing $n_x$, $n_h$ and $n_y$ nodes. A data point $x_i$ contains $n_x$ features at every time-step $t$, sent to the LSTM. The connections between the input and hidden nodes are parametrized by a weight matrix $W_{xh}$ of size $n_x × n_h$. The weights in the hidden layer represent recurrent connections, where connections from hidden layer at time-step $t$ to those at time-step $t + 1$ (i.e. $h_t$ to $h_{t+1}$) are parametrized by a weight matrix $W_{hh}$ of size $n_h × n_h$. Finally, weights from the hidden layer to the output layer at every time-step are parametrized by a weight matrix $W_{hy}$ , of size $n_h × n_y$.

There are some important steps involved in training an LSTM:

1. Forward Pass (Forward Propagation)

   In this step, in general, given an input data point $x = x_1, x_2, ... , x_T$ of $T$ time-steps, the LSTM calculates the output $y = y_1, y_2, ... , y_T$ of $T$ time-steps (this can vary for different applications). All the updates are propagated from $h_1$ to $h_T$.

2. Backward Pass (Backpropagation Through Time)

   In this step, the LSTM do the calculating for optimization method, in this project, I used gradient descent. For this, gradients from time-step $T$ have to be propagated all the way back to time-step 1.
 
3. Parameters Updating

   Now, the LSTM updates the weight matrices using the gradient descent calculated from the previous step.

# Dataset

## Information

Dataset Name: US Airline Passengers

Description:
* The classic Box & Jenkins airline data. Monthly totals of international airline passengers, 1949 to 1960. 
* This dataset provides monthly totals of a US airline passengers from 1949 to 1960. This dataset is taken from an inbuilt dataset of R called AirPassengers.

Kaggle Link: https://www.kaggle.com/datasets/chirag19/air-passengers

Format: `CSV` A monthly time series, in thousands.

Original Source: Box, G. E. P., Jenkins, G. M. and Reinsel, G. C. (1976) Time Series Analysis, Forecasting and Control. Third Edition. Holden-Day. Series G.

More details about the dataset: https://stat.ethz.ch/R-manual/R-devel/library/datasets/html/AirPassengers.html

## EDA & Preprocessing

# Implementation
## Strategy
## Sequential Version
## GPU Parallel Version

# Result

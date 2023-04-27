<h1 align="center">Parallel LSTM training for sequence prediction from sequential data</h1>

<p align="center">Vietnam National University - University of Science - Falculty of Information Technology</p>

<p align="center">CSC14116 - Applied Parallel Programming</p>

<p align="center">19120454 - Bui Quang Bao</p>

# Table of Contents

- [Introduction](#introduction)
- [Background](#background)
  * [About RNN and LSTM](#about-rnn-and-lstm)
  * [LSTM Architecture](#lstm-architecture)
- [Dataset](#dataset)
  * [Information](#information)
  * [Preprocessing](#preprocessing)
- [Implementation](#implementation)
  * [Strategy](#strategy)
  * [Sequential Version](#sequential-version)
  * [Parallel Version 1](#parallel-version-1)
  * [Parallel Version 2](#parallel-version-2)
- [Result](#result)
  * [Comparison](#comparison)
  * [Final Demo](#final-demo)
    + [Light-version Dataset](#light-version-dataset)
    + [Heavy-version Dataset](#heavy-version-dataset)
- [Conclusion](#conclusion)

# Introduction

In this project, I will analyze and parallel the LSTM model (a RNN - Recurrent Neural Network) in order to improve its training speed and efficiency. By utilizing parallel processing and GPU computing, the model will be able to handle larger datasets and have shorter training duration. The specific task that I want to apply using the LSTM model in this project is time-series prediction - sequence prediction from sequential data. I will implement a raw LSTM model using only Numpy library, analyze, parallelize using Numba library, and measure the efficiency of the parallel version over the sequential version.

Keywords: `Parallel Processing`, `GPU Computing`, `NVIDIA CUDA`, `Recurrent Neural Network (RNN)`, `Long Short-Term Memory (LSTM)`

# Background

## About RNN and LSTM

A recurrent neural network (RNN) trains on input containing sequences of data, as it learns about time dependent relations between different parts of the input. For example, if we send as input a sequence of words, i.e. a sentence, an RNN can learn about the relations between different words, and hence learn rules of grammar such as relationships between verbs and adverbs, etc.

LSTM stands for Long Short-Term Memory. It is a type of recurrent neural network (RNN) that is designed to handle the problem of vanishing gradients in traditional RNNs. LSTM networks have the ability to selectively remember or forget information over long periods of time, making them particularly effective for tasks that involve sequential data, such as speech recognition, language translation, and time series prediction.

## LSTM Architecture

A basic LSTM neural network contains an input layer, hidden layer and output layer, containing $n_x$, $n_h$ and $n_y$ nodes. A data point $x_i$ contains $n_x$ features at every time-step $t$, sent to the LSTM. The connections between the input and hidden nodes are parametrized by a weight matrix $W_{xh}$ of size $n_x × n_h$. The weights in the hidden layer represent recurrent connections, where connections from hidden layer at time-step $t$ to those at time-step $t + 1$ (i.e. $h_t$ to $h_{t+1}$) are parametrized by a weight matrix $W_{hh}$ of size $n_h × n_h$. Finally, weights from the hidden layer to the output layer at every time-step are parametrized by a weight matrix $W_{hy}$ , of size $n_h × n_y$.

![Image](/assets/architecture.png)

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

Original Source: *Box, G. E. P., Jenkins, G. M. and Reinsel, G. C. (1976) Time Series Analysis, Forecasting and Control. Third Edition. Holden-Day. Series G.*

More details about the dataset: https://stat.ethz.ch/R-manual/R-devel/library/datasets/html/AirPassengers.html

## Preprocessing

![Image](/visualization/passengers.png)
![Image](/visualization/change.png)
![Image](/visualization/change-normalized.png)
![Image](/visualization/change-encoded.png)

# Implementation

## Strategy

![Image](/assets/strategy.png)

## Sequential Version

```
⚠️ Not done yet ⚠️
```

## Parallel Version 1

```
⚠️ Not done yet ⚠️
```

## Parallel Version 2

```
⚠️ Not done yet ⚠️
```

# Result

## Comparison

Hyper-parameters for the comparison: 
* `num_epochs = 5`
* `learning_rate = 0.1`
* `optimize_method = 'minibatch'`

All the versions have the same outputs/logs (training loss and validation loss) -> Correct implementation.

Logs (same for all versions):

```
Epoch 0:	Train Loss = 3.9	Valid Loss = 4.2 	 (3.9031626016491523 	 4.2027568799857145)
Epoch 1:	Train Loss = 3.62	Valid Loss = 3.86 	 (3.6152784161491525 	 3.8615986598000007)
Epoch 2:	Train Loss = 3.39	Valid Loss = 3.61 	 (3.386136692522034 	 3.6050993745999995)
Epoch 3:	Train Loss = 3.19	Valid Loss = 3.38 	 (3.1897238797101695 	 3.3849550778)
Epoch 4:	Train Loss = 3.02	Valid Loss = 3.2 	 (3.0220866195525424 	 3.198618844957143)
Epoch 5:	Train Loss = 2.88	Valid Loss = 3.04 	 (2.8779136835389827 	 3.0435489866142853)
```

This table compare the running time between sequential and parallel versions using the `%%time` command.

|             | CPU times - user | CPU times - sys | CPU times - total | Wall time | Efficiency | Evaluate     |
|-------------|------------------|-----------------|-------------------|-----------|------------|--------------|
| Sequential  |         4min 29s |          711 ms |          4min 30s |  4min 37s |       100% |              |
| Parallel V1 |         1min 18s |          430 ms |          1min 19s |  1min 21s |       342% |              |
| Parallel V2 |             19 s |          100 ms |            19.1 s |    19.1 s |      1450% | Best Version |

*The `Efficiency` column is the comparison with sequential version.*

## Final Demo

### Light-version Dataset

Hyper-parameters for the demo: 
* `num_epochs = 50`
* `learning_rate = 0.1`
* `optimize_method = 'minibatch'`

Logs:

```
Implementation: parallel_v2
Epoch 0:	Train Loss = 3.9	Valid Loss = 4.2 	 (3.9031626016491523 	 4.2027568799857145)
Epoch 10:	Train Loss = 2.35	Valid Loss = 2.54 	 (2.3531741741525423 	 2.538028188642857)
Epoch 20:	Train Loss = 1.84	Valid Loss = 2.08 	 (1.842756373281356 	 2.0835125372714285)
Epoch 30:	Train Loss = 1.63	Valid Loss = 1.94 	 (1.6328545560355934 	 1.9359360391428573)
Epoch 40:	Train Loss = 1.5	Valid Loss = 1.91 	 (1.4979629829288135 	 1.9138173331857142)
Epoch 50:	Train Loss = 1.39	Valid Loss = 1.9 	 (1.3898027960457626 	 1.8966150371857142)
CPU times: user 1min 34s, sys: 279 ms, total: 1min 35s
Wall time: 1min 36s
```

Result:

![Image](/visualization/demo-light-losses.png)
![Image](/visualization/demo-light-result.png)

### Heavy-version Dataset

```
⚠️ Not done yet ⚠️
```

# Conclusion

```
⚠️ Not done yet ⚠️
```

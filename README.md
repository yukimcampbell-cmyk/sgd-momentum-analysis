# sgd-momentum-analysis
This project investigates and analyzes how different optimization strategies, specifically mini-batching and momentum, affect the training and convergence of a neural network on a non-linear classification problem. I implemented stochastic gradient descent (SGD) with and without momentum and compared full-batch and mini-batch training.

This project also includes an analysis of weight initialization.

NOTE: torch_dataset.py is not provided as it was used in a university course.


## SGD Implementation
I implemented each optimization procedure manually instead of using PyTorch's built-in optimizers.

The algorithm for each batch (sgd function):
1. Performs a forward pass throughout the network
2. Computes the cross-entropy loss
3. Use back propagation to compute gradients
4. Updates each parameter using the SGD update rule

For standard SGD:
[
\theta_{t+1} = \theta_t - \eta \nabla L(\theta_t)
]
where (\eta) is the learning rate.

## Momentum
Momentum was added by maintaining a velocity term for each parameter:
[
v_t = \mu v_{t-1} + \nabla L(\theta_t)
]
[
\theta_{t+1} = \theta_t - \eta v_t
]
where (\mu) is the momentum coefficient.



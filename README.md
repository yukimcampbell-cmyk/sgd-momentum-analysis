# sgd-momentum-analysis
This project investigates and analyzes how different optimization strategies, specifically mini-batching and momentum, affect the training and convergence of a neural network on a non-linear classification problem. I implemented stochastic gradient descent (SGD) with and without momentum and compared full-batch and mini-batch training.

This project also includes an analysis of weight initialization.

NOTE: torch_dataset.py is not provided as it was used in a university course.

## Technologies:
- Python
- NumPy
- PyTorch
- Matplotlib
- tqdm

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

## Experimental Setup
The neural network consisted of: 
- Linear layer: 2 -> 16
- Tanh activation
- Linear layer: 16 -> 16
- Tanh activation
- Linear layer: 16 -> 3

The network was trained on a non-linear Annuli classification dataset with 300 samples and noise level of 0.1 (based on code not provided in this folder).

The four optimization strategies that were compared were:
- Full-batch SGD without momentum
- Mini-batch SGD without momentum
- Full-batch SGD with momentum
- Mini-batch SGD with momentum

For full-batch SGD, a batch size of 300 was used.
For mini-batch SGD, a batch size of 100 was used.
For the two optimization strategies with momentum, a momentum of 0.9 was used.
The learning rate was 0.05 for all strategies.

## Results
The learning curves show how mini-batching and momentum affect the convergence of SGD. There are clear differences in the convergence speed between the four optimization methods: full-batch with no momentum, mini-batches with no momentum, full-batch with momentum, and mini-batches with momentum.

Full-batch without momentum converged the slowest, while adding momentum substationally reduced the number of epochs needed to achieve a low training loss. Mini-batches with momentum converged the fastest, reaching a low loss in approximately 100 epochs.

Overall, the results show that both momentum and mini-batching have a significant effect on the convergence speed of neural network optimization.

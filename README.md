# Variational-Autoencoder-on-Binarized-MNIST-Digits

## Project Overview

Implementation of Variational Autoencoder (VAE) on binarized MNIST digits, based on the paper "Auto-Encoding Variational Bayes" by Kingma and Welling (2013).

## Dataset

- Binarized MNIST dataset (http://yann.lecun.com/exdb/mnist/)
- 10,000 training samples
- 10,000 testing samples
- Optional: 100-sample dataset for debugging

## Implementation

- Variational Autoencoder (VAE) architecture
- Latent space: Gaussian distribution
- Loss function: Evidence Lower Bound (ELBO)

## Tools

- Implemented using Python
- Utilizes automatic differentiation

## Features

- Binarized MNIST dataset loading and preprocessing
- VAE model implementation
- Training and testing scripts
- Hyperparameter tuning

## Requirements

- Python 3.x
- NumPy
- Autograd
- Matplotlib (for visualization)

Usage

1. Clone repository
2. Install requirements
3. Run training script
4. Evaluate model performance on test dataset

References

- Kingma and Welling (2013) - Auto-Encoding Variational Bayes (https://arxiv.org/pdf/1312.6114.pdf)


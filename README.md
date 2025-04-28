# Cartpole PPO — From-Scratch PPO Implementation in PyTorch

This repository contains a vanilla PyTorch implementation of the Proximal Policy Optimization (PPO) algorithm, applied to the classic CartPole environment.

The implementation is done fully from scratch, with no reliance on high-level RL libraries, aiming for clarity and educational value.
Status

The PPO agent successfully solves the CartPole environment. Experimental results will be added as more data becomes available.

There is an existing [checkpoint of a trained agent available on Google Drive](https://drive.google.com/drive/folders/1QfG9LeyMBpxtpwSlvmrwoQ0kw0fpVr55?usp=drive_link)

> ⚠️   Note: The checkpoint is provided for testing and evaluation purposes. Please do not use it for further training.


## Features

- PPO clipped surrogate objective
- Separate actor and critic networks
- Training from scratch, no RL libraries
- Clean, minimal PyTorch codebase

## TODO

- Reward shaping for better performance
- Add detailed experiment results and plots
- Include hyperparameter sweeps
- Extend to other environments (optional)

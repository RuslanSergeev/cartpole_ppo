# Cartpole PPO — From-Scratch PPO Implementation in PyTorch

This repository contains a vanilla PyTorch implementation of the Proximal Policy Optimization (PPO) algorithm, applied to the classic CartPole environment.

The implementation is done fully from scratch, with no reliance on high-level RL libraries, aiming for clarity and educational value.
Status

The PPO agent successfully solves the CartPole environment. Experimental results will be added as more data becomes available.

The best performing model is available in [Google Drive](https://drive.google.com/drive/folders/1QfG9LeyMBpxtpwSlvmrwoQ0kw0fpVr55?usp=drive_link)

> ⚠️   The experiments results are being documented and will be added to this repository later this week.

## How to train a new model

If using pixi, you can run the PPO agent with the following command:
```bash
pixi run -e cpu python3 main.py train <model_path>
```

If using other managers, first activate your environment, then run:
```bash
python3 main.py train <model_path>
```

## How to run a demo

If using pixi, you can run the PPO agent with the following command:
```bash
pixi run -e cpu python3 main.py demo <model_path>
```


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

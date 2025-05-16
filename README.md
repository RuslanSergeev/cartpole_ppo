# CartPole PPO

This repository contains a vanilla PyTorch implementation of the Proximal Policy Optimization (PPO) algorithm, applied to the classic CartPole environment.

The implementation is done fully from scratch, with no reliance on high-level RL libraries, aiming for clarity and educational value.

![Cartpole PPO results]("https://drive.google.com/uc?export=view&id=1bq0X2a3g7v4x5r8j6c9k2m1z4e3f3G5")

## 🔬 Experiments results

The experiments notes and results can be found [here](experiments)

## 🏆 How to train a new model

If using pixi, you can run the PPO agent with the following command:
```bash
pixi run -e cpu python3 main.py train <model_path>
```

If using other managers, first activate your environment, then run:
```bash
python3 main.py train <model_path>
```

## 🏃 How to run a demo

If using pixi, you can run the PPO agent with the following command:
```bash
pixi run -e cpu python3 main.py demo <model_path>
```


## ✨ Features

- PPO clipped surrogate objective
- Separate actor and critic networks
- Training from scratch, no RL libraries
- Clean, minimal PyTorch codebase

## 🌱 TODO

- Reward shaping for better performance
- Add detailed experiment results and plots
- Include hyperparameter sweeps
- Extend to other environments (optional)

## 📚 References

- [PPO algorythms, Arxiv](https://arxiv.org/pdf/1707.06347)
- [GAE - generalized advantage estimation, Arxiv](https://arxiv.org/pdf/1506.02438)
- [Reinforcement Learning: An Introduction](http://incompleteideas.net/book/the-book-2nd.html)
- [Cart-pole dynamics, CTMS](https://ctms.engin.umich.edu/CTMS/index.php?example=InvertedPendulum&section=SystemModeling)

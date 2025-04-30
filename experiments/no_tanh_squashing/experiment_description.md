# No tanh squashing experiment

In this experiment I've completelly disabled actions tanh squashing.  
Git commit hash for the experiment: `8cacab0eabd12c17c2a86991ec937c7503eb1665`  

## Conclusion

The agent is able to maintain a vertical position for a short time (~5 seconds) but is not able to balance the pole for a long time. We need to implement a reward shaping function to improve the performance of the agent. 
- To augment the panalty due to x position and x_dot so that the agent is more stable. 


## Actor network architecture:  

```
Actor:
    Input: 4
    Hidden: 64, relu
    Hidden: 64, relu
    mu: 1, tanh
    log_std: 1, clamp
```

## Critic network architecture:  

```
Critic:
    Input: 4, 64, relu
    Hidden: 64, 64, relu
    V: 64, 1
```

## Loss function:

```
alpha_theta = 1.0
alpha_theta_dot = 0.0
alpha_x = 0.2
alpha_x_dot = 0.0
```

## Hyperparameters:

```
num_episodes=5000,
num_actors=5,
num_epochs=10,
num_time_steps=6000,
lr_actor: float = 1e-4,
lr_critic: float = 0.5e-4,
gamma: float = 0.99,
lam: float = 0.95,
batch_size: int = 32,
```

## Results

```
Episode 665 summ reward: 0.00015997886657714844
Episode 665 average reward: 8.88771456430959e-09
Episode 665 max reward: 1.554492712020874
```

## Assets

![Screencast](https://drive.google.com/file/d/1v8fcEYf9VXLdeCdUKv3ZZFdtJ7vH8QZ_/view?usp=sharing)
[Logs](https://drive.google.com/file/d/1sfWzmVbj71xs75nCpT9PEgX6iNH7Fm8y/view?usp=sharing)



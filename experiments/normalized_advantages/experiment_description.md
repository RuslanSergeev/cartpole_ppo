# No tanh squashing experiment

In this experiment I've disabled reward normalization.
The normalization instead is performed only on advantages.
This is due to the fact that learning the normalized rewards, and associated returns is not feasible for the critic, thus the optimization would never converge.
Git commit hash for the experiment: `a29dd3ce3bb2dae12835c279bfd4f1368b4eacd9`

## Conclusion

The agent is able to maintain the vertical position indefenitely, however it has not learned to approach the target position in the world coordinate system.

## Next steps

- To augment the panalty due to x position, Use the normalized advantages as the base model for subsequent experiments.

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

## Reward function parameters:

```
alpha_theta = 1.0
alpha_theta_dot = 1.0e-5
alpha_x = 1.0e-2
alpha_x_dot = 1.0e-5
```

## Hyperparameters:

```
num_episodes=5000,
num_actors=20,
num_epochs=10,
num_time_steps=6000,
lr_actor: float = 1e-5,
lr_critic: float = 0.5e-4,
gamma: float = 0.99,
lam: float = 0.95,
batch_size: int = 64,
```

## Results

```
Average reward: 0.9857727885246277
Max reward: 0.9999441504478455
```

## Assets

![Screencast](https://drive.google.com/uc?export=view&id=1yyTquiBVBR-FroSPXF2pkwb3ID52YVp1)
![Training curve](https://drive.google.com/uc?export=view&id=1NWsUjjLtYr2XbjHCa4PyVcCddQBvBKEm)
[Logs](https://drive.google.com/file/d/1TihEnzzaqt8vpBXcCeDLVGJQ3v6XeozK/view?usp=drive_link)  
[Model checkpoint](https://drive.google.com/file/d/1dggc1lkngiCZiMt0I3060wVkuzOW8ARp/view?usp=drive_link)


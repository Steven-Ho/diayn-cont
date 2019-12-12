# Continuous Latent Space for DIAYN

## What is DIAYN?

DIAYN is a variational approach for reinforcement learning aiming to discover diverse skills without external rewards. The paper is here: [Diversity is All You Need: Learning Skills without a Reward Function](https://arxiv.org/abs/1802.06070). And this is an implementation by the authors: [Diversity Is All You Need](https://github.com/haarnoja/sac/blob/master/DIAYN.md). This project is based on SAC (Soft Actor-Critic), the maximum-entropy RL.

## What is this project?

This project is another version of DIAYN but different on these two aspects:

1. The latent space is continuous, and the prior is a uniform distribution over [-1, 1].
2. The algorithm is implemented on PyTorch and Gym.

With a continuous latent space, DIAYN may learn a spectrum of policies rather than several independent skills. Besides, this project gets rid of rllab, which is not under maintaining.

## How to use it?

The dependencies are `gym` and `PyTorch`.

Just run `python main.py`. The options of training is listed is `main.py` file. It will automatically record the training process and can be viewed in TensorBoard.

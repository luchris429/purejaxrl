# PureJaxRL (End-to-End RL Training in Pure Jax)

[<img src="https://img.shields.io/badge/license-Apache2.0-blue.svg">](https://github.com/luchris429/purejaxrl/LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/luchris429/purejaxrl/blob/main/examples/walkthrough.ipynb)

PureJaxRL is a high-performance, end-to-end Jax Reinforcement Learning (RL) implementation. When running many agents in parallel on GPUs, our implementation is over 1000x faster than standard PyTorch RL implementations. Unlike other Jax RL implementations, we implement the *entire training pipeline in JAX*, including the environment. This allows us to get significant speedups through JIT compilation and by avoiding CPU-GPU data transfer. It also results in easier debugging because the system is fully synchronous. More importantly, this code allows you to use jax to `jit`, `vmap`, `pmap`, and `scan` entire RL training pipelines. With this, we can:

- 🏃 Efficiently run tons of seeds in parallel on one GPU
- 💻 Perform rapid hyperparameter tuning
- 🦎 Discover new RL algorithms with meta-evolution

For more details, visit the accompanying blog post: https://chrislu.page/blog/meta-disco/

This notebook walks through the basic usage: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/luchris429/purejaxrl/blob/main/examples/walkthrough.ipynb)

## Performance

Without vectorization, our implementation runs 10x faster than [CleanRL's PyTorch baselines](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo.py), as shown in the single-thread performance plot.

Cartpole                   |  Minatar-Breakout
:-------------------------:|:-------------------------:
![](docs/cartpole_plot_seconds.png)  |  ![](docs/minatar_plot_seconds.png)


With vectorized training, we can train 2048 PPO agents in half the time it takes to train a single PyTorch PPO agent on a single GPU. The vectorized agent training allows for simultaneous training across multiple seeds, rapid hyperparameter tuning, and even evolutionary Meta-RL. 

Vectorised Cartpole        |  Vectorised Minatar-Breakout
:-------------------------:|:-------------------------:
![](docs/cartpole_plot_parallel.png)  |  ![](docs/minatar_plot_parallel.png)


## Code Philosophy

PureJaxRL is inspired by [CleanRL](https://github.com/vwxyzjn/cleanrl), providing high-quality single-file implementations with research-friendly features. Like CleanRL, this is not a modular library and is not meant to be imported. The repository focuses on simplicity and clarity in its implementations, making it an excellent resource for researchers and practitioners.

## Installation

Install dependencies using the requirements.txt file:

```
pip install -r requirements.txt
```

In order to use JAX on your accelerators, you can find more details in the [JAX documentation](https://github.com/google/jax#installation).

## Example Usage

[`examples/walkthrough.ipynb`](https://github.com/luchris429/purejaxrl/blob/main/examples/walkthrough.ipynb) walks through the basic usage. [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/luchris429/purejaxrl/blob/main/examples/walkthrough.ipynb)

[`examples/brax_minatar.ipynb`](https://github.com/luchris429/purejaxrl/blob/main/examples/brax_minatar.ipynb) walks through using PureJaxRL for Brax and MinAtar. [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/luchris429/purejaxrl/blob/main/examples/brax_minatar.ipynb)

## TODOs

The following improvements are planned for the PureJaxRL repository:

1. More memory-efficient logging
2. Integration with Weights & Biases (WandB) for experiment tracking
3. Connecting to non-Jax environments like envpool

## Related Work

PureJaxRL builds upon other tools in the Jax and RL ecosystems. Check out the following projects:

- Gymnax (https://github.com/RobertTLange/gymnax)
- Evosax (https://github.com/RobertTLange/evosax)
- CleanRL (https://github.com/vwxyzjn/cleanrl)
- Brax (https://github.com/google/brax)
- Jumanji (https://github.com/instadeepai/jumanji)
- Pgx (https://github.com/sotetsuk/pgx)

The following repositories and projects were pre-cursors to `purejaxrl`:

- [Model-Free Opponent Shaping](https://arxiv.org/abs/2205.01447) (ICML 2022) (https://github.com/luchris429/Model-Free-Opponent-Shaping)

- [Discovered Policy Optimisation](https://arxiv.org/abs/2210.05639) (NeurIPS 2022) (https://github.com/luchris429/discovered-policy-optimisation)

- [Adversarial Cheap Talk](https://arxiv.org/abs/2211.11030) (ICML 2023) (https://github.com/luchris429/adversarial-cheap-talk)

## Citation

If you use PureJaxRL in your work, please cite the following paper:

```
@article{lu2022discovered,
    title={Discovered policy optimisation},
    author={Lu, Chris and Kuba, Jakub and Letcher, Alistair and Metz, Luke and Schroeder de Witt, Christian and Foerster, Jakob},
    journal={Advances in Neural Information Processing Systems},
    volume={35},
    pages={16455--16468},
    year={2022}
}
```

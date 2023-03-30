# End-to-End Accelerated Reinforcement Learning Algortihms in Jax

[<img src="https://img.shields.io/badge/license-MIT-blue">](https://github.com/luchris429/FastJaxRL)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/luchris429/FastJaxRL/examples/example_0.ipynb)

FastJaxRL is a high-performance, end-to-end Jax Reinforcement Learning (RL) implementation, including the environment. By running entirely on the accelerator, we leverage Jax's vectorization capabilities and overcome the bottlenecks of CPU-GPU data transfer and Python overhead. This results in significant speedups, easier debugging, and fully synchronous operation. For more details, visit the accompanying blog post: https://chrislu.page/blog/meta-disco/

## Performance

Our implementation runs 10x faster than CleanRL's PyTorch baselines, as shown in the single-thread performance plot.

Cartpole                   |  Minatar-Breakout
:-------------------------:|:-------------------------:
![](docs/cartpole_plot_seconds.png)  |  ![](docs/minatar_plot_seconds.png)


The vectorized agent training allows for simultaneous training across multiple seeds, rapid hyperparameter tuning, and even evolutionary Meta-RL. With vectorized training, we can train 2048 PPO agents in half the time it takes to train a single PyTorch PPO agent.

Cartpole                   |  Minatar-Breakout
:-------------------------:|:-------------------------:
![](docs/cartpole_plot_parallel.png)  |  ![](docs/minatar_plot_parallel.png)


## Code Philosophy

FastJaxRL is inspired by CleanRL, providing high-quality single-file implementations with research-friendly features. Note that it is not a modular library and is not meant to be imported. The repository focuses on simplicity and clarity in its implementations, making it an excellent resource for researchers and practitioners alike.

## Installation

Install dependencies using the requirements.txt file:

pip install -r requirements.txt

## Example Usage

Check out the example notebook examples.ipynb for basic usage of FastJaxRL. You can open it in Google Colab to start experimenting.

## TODOs

The following improvements are planned for the FastJaxRL repository:

1. More memory-efficient logging
2. Integration with Weights & Biases (WandB) for experiment tracking
3. Continuous PPO for Brax
5. Connecting to non-Jax environments like envpool

## Related Work

FastJaxRL builds upon other tools in the Jax ecosystem. Check out the following projects:

- Gymnax (https://github.com/RobertTLange/gymnax)
- Evosax (https://github.com/RobertTLange/evosax)
- Brax (https://github.com/google/brax)
- Jumanji (https://github.com/instadeepai/jumanji)

## Citation

If you use FastJaxRL in your work, please cite the following paper:
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
# PureJaxRL Resources 

Last year, I released [PureJaxRL](https://github.com/luchris429/purejaxrl), a simple repository that implements RL algorithms entirely end-to-end in JAX, which enables speedups of up to 4000x in RL training. PureJaxRL, in turn, was inspired by multiple projects, including [CleanRL](https://github.com/vwxyzjn/cleanrl) and [Gymnax](https://github.com/RobertTLange/gymnax). Since the release of PureJaxRL, a large number of projects related to or inspired by PureJaxRL have come out, vastly expanding its use case from standard single-agent RL settings. This curated list contains those projects alongside other relevant implementations of algorithms, environments, tools, and tutorials.

To understand more about the benefits PureJaxRL, I recommend viewing the [original blog post](https://chrislu.page/blog/meta-disco/) or [tweet thread](https://x.com/_chris_lu_/status/1643992216413831171).

The PureJaxRL repository can be found here:

[https://github.com/luchris429/purejaxrl/](https://github.com/luchris429/purejaxrl/). <img src="https://img.shields.io/github/stars/luchris429/purejaxrl?style=social" align="center">

The format of the list is from [awesome](https://github.com/sindresorhus/awesome) and [awesome-jax](https://github.com/n2cholas/awesome-jax). While this list is curated, it is certainly not complete. If you have a repository you would like to add, please contribute!

If you find this resource useful, please *star* the repo! It helps establish and grow the end-to-end JAX RL community.

## Contents

- [Algorithms](#algorithms)
- [Environments](#environments)
- [Related Components](#components)
- [Tutorials and Blog Posts](#tutorials-and-blog-posts)
- [Related Papers](#papers)

## Algorithms

### End-to-End JAX RL Implementations

- [purejaxrl](https://github.com/luchris429/purejaxrl) - Classic and simple end-to-end RL training in pure JAX. <img src="https://img.shields.io/github/stars/luchris429/purejaxrl?style=social" align="center">

- [rejax](https://github.com/keraJLi/rejax) - Modular and importable end-to-end JAX RL training. <img src="https://img.shields.io/github/stars/keraJLi/rejax?style=social" align="center">

- [Stoix](https://github.com/EdanToledo/Stoix) - End-to-end JAX RL training with advanced logging, configs, and more. <img src="https://img.shields.io/github/stars/EdanToledo/Stoix?style=social" align="center">

- [purejaxql](https://github.com/mttga/purejaxql/) - Simple single-file end-to-end JAX baselines for Q-Learning. <img src="https://img.shields.io/github/stars/mttga/purejaxql?style=social" align="center">

- [jym](https://github.com/rpegoud/jym) - Educational and beginner-friendly end-to-end JAX RL training. <img src="https://img.shields.io/github/stars/rpegoud/jym?style=social" align="center">

### Jax RL (But Not End-to-End) Repos

- [cleanrl](https://github.com/vwxyzjn/cleanrl) - Clean implementations of RL Algorithms (in both PyTorch and JAX!). <img src="https://img.shields.io/github/stars/vwxyzjn/cleanrl?style=social" align="center">

- [jaxrl](https://github.com/ikostrikov/jaxrl) - JAX implementation of algorithms for Deep Reinforcement Learning with continuous action spaces. <img src="https://img.shields.io/github/stars/ikostrikov/jaxrl?style=social" align="center">

- [rlbase](https://github.com/kvfrans/rlbase_stable) - Single-file JAX implementations of Deep RL algorithms. <img src="https://img.shields.io/github/stars/kvfrans/rlbase_stable?style=social" align="center">

### Multi-Agent RL

- [JaxMARL](https://github.com/FLAIROx/JaxMARL) - Multi-Agent RL Algorithms and Environments in pure JAX. <img src="https://img.shields.io/github/stars/FLAIROx/JaxMARL?style=social" align="center">

- [Mava](https://github.com/instadeepai/Mava) - Multi-Agent RL Algorithms in pure JAX (previously tensorflow-based algorithms). <img src="https://img.shields.io/github/stars/instadeepai/Mava?style=social" align="center">

- [pax](https://github.com/ucl-dark/pax) - Scalable Opponent Shaping Algorithms in pure JAX. <img src="https://img.shields.io/github/stars/ucl-dark/pax?style=social" align="center">

### Offline RL

- [JAX-CORL](https://github.com/nissymori/JAX-CORL) - Single-file implementations of offline RL algorithms in JAX. <img src="https://img.shields.io/github/stars/nissymori/JAX-CORL?style=social" align="center">

### Inverse-RL

- [jaxirl](https://github.com/FLAIROx/jaxirl) - Pure JAX for Inverse Reinforcement Learning. <img src="https://img.shields.io/github/stars/FLAIROx/jaxirl?style=social" align="center">

### Unsupervised Environment Design

- [minimax](https://github.com/facebookresearch/minimax) - Canonical implementations of UED algorithms in pure JAX, including SSM-based acceleration. <img src="https://img.shields.io/github/stars/facebookresearch/minimax?style=social" align="center">

- [jaxued](https://github.com/DramaCow/jaxued) - Single-file implementations of UED algorithms in pure JAX. <img src="https://img.shields.io/github/stars/DramaCow/jaxued?style=social" align="center">

### Quality-Diversity

- [QDax](https://github.com/adaptive-intelligent-robotics/QDax) - Quality-Diversity algorithms in pure JAX. <img src="https://img.shields.io/github/stars/adaptive-intelligent-robotics/QDax?style=social" align="center">

### Partially-Observed RL

- [popjaxrl](https://github.com/luchris429/popjaxrl) - Partially-observed RL environments (POPGym) and architectures (incl. SSM's) in pure JAX. <img src="https://img.shields.io/github/stars/luchris429/popjaxrl?style=social" align="center">

### Meta-Learning RL Objectives

- [groove](https://github.com/EmptyJackson/groove) - Library for [LPG-like](https://arxiv.org/abs/2007.08794) meta-RL in Pure JAX. <img src="https://img.shields.io/github/stars/EmptyJackson/groove?style=social" align="center">

- [discovered-policy-optimisation](https://github.com/luchris429/discovered-policy-optimisation) - Library for [LPO](https://arxiv.org/abs/2210.05639) meta-RL in Pure JAX. <img src="https://img.shields.io/github/stars/luchris429/discovered-policy-optimisation?style=social" align="center">

- [rl-learned-optimization](https://github.com/AlexGoldie/rl-learned-optimization) - Library for [OPEN](https://arxiv.org/abs/2407.07082) in Pure JAX. <img src="https://img.shields.io/github/stars/AlexGoldie/rl-learned-optimization?style=social" align="center">

## Environments

- [gymnax](https://github.com/RobertTLange/gymnax) - Classic RL environments in JAX. <img src="https://img.shields.io/github/stars/RobertTLange/gymnax?style=social" align="center">

- [brax](https://github.com/google/brax) - Continuous control environments in JAX. <img src="https://img.shields.io/github/stars/google/brax?style=social" align="center">

- [JaxMARL](https://github.com/FLAIROx/JaxMARL) - Multi-agent algorithms and environments in pure JAX. <img src="https://img.shields.io/github/stars/FLAIROx/JaxMARL?style=social" align="center">

- [jumanji](https://github.com/instadeepai/jumanji) - Suite of unique RL environments in JAX. <img src="https://img.shields.io/github/stars/instadeepai/jumanji?style=social" align="center">

- [pgx](https://github.com/sotetsuk/pgx) - Suite of popular board games in JAX. <img src="https://img.shields.io/github/stars/sotetsuk/pgx?style=social" align="center">

- [popjaxrl](https://github.com/luchris429/popjaxrl) - Partially-observed RL environments (POPGym) in JAX. <img src="https://img.shields.io/github/stars/luchris429/popjaxrl?style=social" align="center">

- [waymax](https://github.com/waymo-research/waymax) - Self-driving car simulator in JAX. <img src="https://img.shields.io/github/stars/waymo-research/waymax?style=social" align="center">

- [Craftax](https://github.com/MichaelTMatthews/Craftax) - A challenging crafter-like and nethack-inspired benchmark in JAX. <img src="https://img.shields.io/github/stars/MichaelTMatthews/Craftax?style=social" align="center">

- [xland-minigrid](https://github.com/corl-team/xland-minigrid) - A large-scale meta-RL environment in JAX. <img src="https://img.shields.io/github/stars/corl-team/xland-minigrid?style=social" align="center">

- [navix](https://github.com/epignatelli/navix) - Classic minigrid environments in JAX. <img src="https://img.shields.io/github/stars/epignatelli/navix?style=social" align="center">

- [autoverse](https://github.com/smearle/autoverse) - A fast, evolvable description language for reinforcement learning environments. <img src="https://img.shields.io/github/stars/smearle/autoverse?style=social" align="center">

- [qdx](https://github.com/jolle-ag/qdx) - Quantum Error Corection with JAX. <img src="https://img.shields.io/github/stars/jolle-ag/qdx?style=social" align="center">

- [matrax](https://github.com/instadeepai/matrax) - Matrix games in JAX. <img src="https://img.shields.io/github/stars/instadeepai/matrax?style=social" align="center">

- [AlphaTrade](https://github.com/KangOxford/AlphaTrade) - Limit Order Book (LOB) in JAX. <img src="https://img.shields.io/github/stars/KangOxford/AlphaTrade?style=social" align="center">

## Relevant Tools and Components

- [evosax](https://github.com/RobertTLange/evosax) - Evolution strategies in JAX. <img src="https://img.shields.io/github/stars/RobertTLange/evosax?style=social" align="center">

- [evojax](https://github.com/google/evojax) - Evolution strategies in JAX. <img src="https://img.shields.io/github/stars/google/evojax?style=social" align="center">

- [flashbax](https://github.com/instadeepai/flashbax) - Accelerated replay buffers in JAX. <img src="https://img.shields.io/github/stars/instadeepai/flashbax?style=social" align="center">

- [dejax](https://github.com/hr0nix/dejax) - Accelerated replay buffers in JAX. <img src="https://img.shields.io/github/stars/hr0nix/dejax?style=social" align="center">

- [rlax](https://github.com/google-deepmind/rlax) - RL components and building blocks in JAX. <img src="https://img.shields.io/github/stars/google-deepmind/rlax?style=social" align="center">

- [mctx](https://github.com/google-deepmind/mctx) - Monte Carlo tree searh in JAX. <img src="https://img.shields.io/github/stars/google-deepmind/mctx?style=social" align="center">

- [distrax](https://github.com/google-deepmind/distrax) - Distributions in JAX. <img src="https://img.shields.io/github/stars/google-deepmind/distrax?style=social" align="center">

- [optax](https://github.com/google-deepmind/optax) - Gradient-based optimizers in JAX. <img src="https://img.shields.io/github/stars/google-deepmind/optax?style=social" align="center">

- [flax](https://github.com/google/flax) - Neural Networks in JAX. <img src="https://img.shields.io/github/stars/google/flax?style=social" align="center">

## Tutorials and Blog Posts

- [Achieving 4000x Speedups with PureJaxRL](https://chrislu.page/blog/meta-disco/) - A blog post on how JAX can massively speedup RL training through vectorisation.

- [Breaking down State-of-the-Art PPO Implementations in JAX](https://towardsdatascience.com/breaking-down-state-of-the-art-ppo-implementations-in-jax-6f102c06c149) - A blog post explaining PureJaxRL's PPO Implementation in depth.

- [A Gentle Introduction to Deep Reinforcement Learning in JAX](https://towardsdatascience.com/a-gentle-introduction-to-deep-reinforcement-learning-in-jax-c1e45a179b92) - A JAX tutorial on Deep RL.

- [Writing an RL Environment in JAX](https://medium.com/@ngoodger_7766/writing-an-rl-environment-in-jax-9f74338898ba) - A JAX tutorial on making environments.

- [Getting started with JAX (MLPs, CNNs & RNNs)](https://roberttlange.com/posts/2020/03/blog-post-10/) - A basic JAX neural network tutorial.

- [awesome-jax](https://github.com/n2cholas/awesome-jax) - A list of useful libraries in JAX

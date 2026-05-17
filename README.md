# PI-CViT

This repository contains the code for the paper "On training of physics-informed neural operator for solving parametric PDEs". 

## Overview

<img src="img/masterfigure.png" alt="masterfigure" width="720"/>


Error distribution of different models on each benchmark problem:

<img src="img/error_distribution.png" alt="error_distribution" width="720"/>

Model performance with different data regimes:

<img src="img/sampling_regimes.png" alt="data_regimes" width="600"/>

<img src="img/data_vs_physics.png" alt="data_vs_physics" width="600"/>


## Quick Start

Taking `Burgers' Equation` as an example, we provide the quick start guide for training `PI-CViT` on `Burgers' Equation`.

To generate the reference solution for `Burgers' Equation`, configure the parameters in `burgers/solver_jax.py` and run the following command in the top level directory of the repository:
```bash
python burgers/solver_jax.py
```

This will generate the reference solution and save it in `./data/burgers/`. Then train different models on `Burgers' Equation` with the following commands:
```bash
python -m burgers.train --configs train_cvit --set save_dir=<YOUR_LOG_DIR> --set data_dir=./data/burgers
python -m burgers.train --configs train_deeponet --set save_dir=<YOUR_LOG_DIR> --set data_dir=./data/burgers
python -m burgers.train_pifno --configs train_fno --set save_dir=<YOUR_LOG_DIR> --set data_dir=./data/burgers
```
For other configurations, please refer to `burgers/configs/`.

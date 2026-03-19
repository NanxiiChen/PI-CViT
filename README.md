# PI-CViT: Physics-Informed Continuous Vision Transformer for Solving Parametric PDEs


## Usage

Train `cvit` on the Burgers' equation:

```bash
python -m burgers.train --configs train_cvit
```

Train `deeponet` on the Burgers' equation:

```bash
python -m burgers.train --configs train_deeponet
```

Train `fno` on the Burgers' equation:

```bash
python -m burgers.train_pifno --configs train_fno
```
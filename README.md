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

Train `fno` on those equations:

```bash
python -m burgers.train_pifno --configs train_fno --optimizer adam --save_dir ./logs/burgers/fno
python -m wave.train_pifno --configs train_fno --optimizer adam --save_dir ./logs/wave/fno
python -m swe.train_pifno --configs train_fno --optimizer adam --save_dir ./logs/swe/fno
python -m swe.train_pifno --configs train_fno --optimizer soap --save_dir ./logs/swe/fno
python -m ldc.train_pifno --configs train_fno --optimizer adam --save_dir ./logs/ldc/fno
```
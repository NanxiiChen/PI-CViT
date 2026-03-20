# PI-CViT: Physics-Informed Continuous Vision Transformer for Solving Parametric PDEs


## Ablation Study

### GradNorm

Train `cvit` without GradNorm:

```bash
python -m burgers.train --configs train_cvit --set use_gradnorm=False --set save_dir=./logs/burgers/cvit/no_gradnorm
python -m ice_melting.train --configs train_cvit --set use_gradnorm=False --set save_dir=./logs/ice_melting/cvit/no_gradnorm
python -m wave.train --configs train_cvit --set use_gradnorm=False --set save_dir=./logs/wave/cvit/no_gradnorm
python -m swe.train --configs train_cvit --set use_gradnorm=False --set save_dir=./logs/swe/cvit/no_gradnorm
python -m ldc.train --configs train_cvit --set use_gradnorm=False --set save_dir=./logs/ldc/cvit/no_gradnorm
```

### Causality

Train `cvit` without causal:

```bash
python -m burgers.train --configs train_cvit --set use_causality=False --set save_dir=./logs/burgers/cvit/no_causality
python -m ice_melting.train --configs train_cvit --set use_causality=False --set save_dir=./logs/ice_melting/cvit/no_causality
python -m wave.train --configs train_cvit --set use_causality=False --set save_dir=./logs/wave/cvit/no_causality
python -m swe.train --configs train_cvit --set use_causality=False --set save_dir=./logs/swe/cvit/no_causality
python -m ldc.train --configs train_cvit --set use_causality=False --set save_dir=./logs/ldc/cvit/no_causality
```
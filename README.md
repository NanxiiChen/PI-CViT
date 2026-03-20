# PI-CViT: Physics-Informed Continuous Vision Transformer for Solving Parametric PDEs


## Ablation Study

### GradNorm

Train `cvit` without GradNorm:

```bash
python -m burgers.train \
    --configs train_cvit \
    --set use_gradnorm=False \
    --set save_dir=./logs/burgers/cvit/no_gradnorm \
    --set save_every=-1 \
    --set test_every=200 \
    --set log_every=50

python -m ice_melting.train \
    --configs train_cvit \
    --set use_gradnorm=False \
    --set save_dir=./logs/ice_melting/cvit/no_gradnorm \
    --set save_every=-1 \
    --set test_every=200 \
    --set log_every=50

python -m wave.train \
    --configs train_cvit \
    --set use_gradnorm=False \
    --set save_dir=./logs/wave/cvit/no_gradnorm \
    --set save_every=-1 \
    --set test_every=200 \
    --set log_every=50

python -m swe.train \
    --configs train_cvit \
    --set use_gradnorm=False \
    --set save_dir=./logs/swe/cvit/no_gradnorm \
    --set save_every=-1
    --set test_every=200
    --set log_every=50

python -m ldc.train \
    --configs train_cvit \
    --set use_gradnorm=False \
    --set save_dir=./logs/ldc/cvit/no_gradnorm \
    --set save_every=-1 \
    --set test_every=200 \
    --set log_every=50
```
`save_every=-1` means that the model weights will not be saved during training.

### Causality

Train `cvit` without causal:

```bash
python -m burgers.train \
    --configs train_cvit \
    --set use_causality=False \
    --set save_dir=./logs/burgers/cvit/no_causality \
    --set save_every=-1 \
    --set test_every=200 \
    --set log_every=50

python -m ice_melting.train \
    --configs train_cvit \
    --set use_causality=False \
    --set save_dir=./logs/ice_melting/cvit/no_causality \
    --set save_every=-1 \
    --set test_every=200 \
    --set log_every=50

python -m wave.train \
    --configs train_cvit \
    --set use_causality=False \
    --set save_dir=./logs/wave/cvit/no_causality \
    --set save_every=-1 \
    --set test_every=200 \
    --set log_every=50

python -m swe.train \
    --configs train_cvit \
    --set use_causality=False \
    --set save_dir=./logs/swe/cvit/no_causality \
    --set save_every=-1 \
    --set test_every=200 \
    --set log_every=50

python -m ldc.train \
    --configs train_cvit \
    --set use_causality=False \
    --set save_dir=./logs/ldc/cvit/no_causality \
    --set save_every=-1 \
    --set test_every=200 \
    --set log_every=50
```

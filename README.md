# PI-CViT: Physics-Informed Continuous Vision Transformer for Solving Parametric PDEs


## Ablation Study

### GradNorm

Train `FNO` without `GradNorm`:

```bash
python -m wave.train_pifno \
    --configs train_fno \
    --set use_gradnorm=False \
    --set save_every=-1 \
    --set log_every=50 \
    --set test_every=200 \
    --set save_dir=./logs/wave/fno/no_gradnorm

python -m swe.train_pifno \
    --configs train_fno \
    --set use_gradnorm=False \
    --set save_every=-1 \
    --set log_every=50 \
    --set test_every=200 \
    --set save_dir=./logs/swe/fno/no_gradnorm
```

Train `FNO` without `Causal`:

```bash
python -m burgers.train_pifno \
    --configs train_fno \
    --set use_causality=False \
    --set save_every=-1 \
    --set log_every=50 \
    --set test_every=200 \
    --set save_dir=./logs/burgers/fno/no_causal 

python -m ice_melting.train_pifno \
    --configs train_fno \
    --set use_causality=False \
    --set save_every=-1 \
    --set log_every=50 \
    --set test_every=200 \
    --set save_dir=./logs/ice_melting/fno/no_causal

python -m wave.train_pifno \
    --configs train_fno \
    --set use_causality=False \
    --set save_every=-1 \
    --set log_every=50 \
    --set test_every=200 \
    --set save_dir=./logs/wave/fno/no_causal

python -m swe.train_pifno \
    --configs train_fno \
    --set use_causality=False \
    --set save_every=-1 \
    --set log_every=50 \
    --set test_every=200 \
    --set save_dir=./logs/swe/fno/no_causal
```
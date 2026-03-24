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

### Number of training data series

#### Shallow Water Equations

First generate the dataset by running the following command in the top level directory of the repository:
```bash
python swe/solver_jax.py
```
This scripts will generate 1024 training data series. The dataset is saved in `./data/swe/f10/swe_training.npz` by default. The size of the training dataset is about 4.8GB.


Train with pure data:

```bash
python -m swe.train_data_driven --configs train_data --set active_loss_names="('data', 'ic_h', 'ic_uv')" --set dataset_size=1024 --set save_name=dataset_size_1024 --set save_dir=./logs/swe/cvit/pure_data/ 
python -m swe.train_data_driven --configs train_data --set active_loss_names="('data', 'ic_h', 'ic_uv')" --set dataset_size=512 --set save_name=dataset_size_512 --set save_dir=./logs/swe/cvit/pure_data/ 
python -m swe.train_data_driven --configs train_data --set active_loss_names="('data', 'ic_h', 'ic_uv')" --set dataset_size=256 --set save_name=dataset_size_256 --set save_dir=./logs/swe/cvit/pure_data/
python -m swe.train_data_driven --configs train_data --set active_loss_names="('data', 'ic_h', 'ic_uv')" --set dataset_size=128 --set save_name=dataset_size_128 --set save_dir=./logs/swe/cvit/pure_data/
python -m swe.train_data_driven --configs train_data --set active_loss_names="('data', 'ic_h', 'ic_uv')" --set dataset_size=64 --set save_name=dataset_size_64 --set save_dir=./logs/swe/cvit/pure_data/
python -m swe.train_data_driven --configs train_data --set active_loss_names="('data', 'ic_h', 'ic_uv')" --set dataset_size=32 --set save_name=dataset_size_32 --set save_dir=./logs/swe/cvit/pure_data/
```

Train with physics on unlabled data:

```bash
python -m swe.train_data_driven --configs train_data --set active_loss_names="('data', 'momentum', 'continuity', 'ic_h', 'ic_uv')" --set dataset_size=1024 --set physics_on_data=False --set save_name=dataset_size_1024 --set save_dir=./logs/swe/cvit/physics_on_unlabeled_data/
python -m swe.train_data_driven --configs train_data --set active_loss_names="('data', 'momentum', 'continuity', 'ic_h', 'ic_uv')" --set dataset_size=512 --set physics_on_data=False --set save_name=dataset_size_512 --set save_dir=./logs/swe/cvit/physics_on_unlabeled_data/
python -m swe.train_data_driven --configs train_data --set active_loss_names="('data', 'momentum', 'continuity', 'ic_h', 'ic_uv')" --set dataset_size=256 --set physics_on_data=False --set save_name=dataset_size_256 --set save_dir=./logs/swe/cvit/physics_on_unlabeled_data/
python -m swe.train_data_driven --configs train_data --set active_loss_names="('data', 'momentum', 'continuity', 'ic_h', 'ic_uv')" --set dataset_size=128 --set physics_on_data=False --set save_name=dataset_size_128 --set save_dir=./logs/swe/cvit/physics_on_unlabeled_data/
python -m swe.train_data_driven --configs train_data --set active_loss_names="('data', 'momentum', 'continuity', 'ic_h', 'ic_uv')" --set dataset_size=64 --set physics_on_data=False --set save_name=dataset_size_64 --set save_dir=./logs/swe/cvit/physics_on_unlabeled_data/
python -m swe.train_data_driven --configs train_data --set active_loss_names="('data', 'momentum', 'continuity', 'ic_h', 'ic_uv')" --set dataset_size=32 --set physics_on_data=False --set save_name=dataset_size_32 --set save_dir=./logs/swe/cvit/physics_on_unlabeled_data/
```

Train with physics on labeled data:

```bash
python -m swe.train_data_driven --configs train_data --set active_loss_names="('data', 'momentum', 'continuity', 'ic_h', 'ic_uv')" --set dataset_size=1024 --set physics_on_data=True --set save_name=dataset_size_1024 --set save_dir=./logs/swe/cvit/physics_on_labeled_data/
python -m swe.train_data_driven --configs train_data --set active_loss_names="('data', 'momentum', 'continuity', 'ic_h', 'ic_uv')" --set dataset_size=512 --set physics_on_data=True --set save_name=dataset_size_512 --set save_dir=./logs/swe/cvit/physics_on_labeled_data/
python -m swe.train_data_driven --configs train_data --set active_loss_names="('data', 'momentum', 'continuity', 'ic_h', 'ic_uv')" --set dataset_size=256 --set physics_on_data=True --set save_name=dataset_size_256 --set save_dir=./logs/swe/cvit/physics_on_labeled_data/
python -m swe.train_data_driven --configs train_data --set active_loss_names="('data', 'momentum', 'continuity', 'ic_h', 'ic_uv')" --set dataset_size=128 --set physics_on_data=True --set save_name=dataset_size_128 --set save_dir=./logs/swe/cvit/physics_on_labeled_data/
python -m swe.train_data_driven --configs train_data --set active_loss_names="('data', 'momentum', 'continuity', 'ic_h', 'ic_uv')" --set dataset_size=64 --set physics_on_data=True --set save_name=dataset_size_64 --set save_dir=./logs/swe/cvit/physics_on_labeled_data/
python -m swe.train_data_driven --configs train_data --set active_loss_names="('data', 'momentum', 'continuity', 'ic_h', 'ic_uv')" --set dataset_size=32 --set physics_on_data=True --set save_name=dataset_size_32 --set save_dir=./logs/swe/cvit/physics_on_labeled_data/
```


#### Lid-Driven Cavity Flow

Put the dataset `ldc_training.npz` in `./data/ldc/`, which contains 256 training data series. 

Train with pure data:

```bash
python -m ldc.train_data_driven --configs train_data --set active_loss_names="('data', 'bc_walls', 'bc_lid', 'bc_pressure')" --set dataset_size=256 --set save_name=dataset_size_256 --set save_dir=./logs/ldc/cvit/pure_data/
python -m ldc.train_data_driven --configs train_data --set active_loss_names="('data', 'bc_walls', 'bc_lid', 'bc_pressure')" --set dataset_size=128 --set save_name=dataset_size_128 --set save_dir=./logs/ldc/cvit/pure_data/
python -m ldc.train_data_driven --configs train_data --set active_loss_names="('data', 'bc_walls', 'bc_lid', 'bc_pressure')" --set dataset_size=64 --set save_name=dataset_size_64 --set save_dir=./logs/ldc/cvit/pure_data/
python -m ldc.train_data_driven --configs train_data --set active_loss_names="('data', 'bc_walls', 'bc_lid', 'bc_pressure')" --set dataset_size=32 --set save_name=dataset_size_32 --set save_dir=./logs/ldc/cvit/pure_data/
```

Train with physics on unlabeled data:

```bash
python -m ldc.train_data_driven --configs train_data --set active_loss_names="('data', 'momentum', 'continuity', 'bc_walls', 'bc_lid', 'bc_pressure')" --set dataset_size=256 --set physics_on_data=False --set save_name=dataset_size_256 --set save_dir=./logs/ldc/cvit/physics_on_unlabeled_data/
python -m ldc.train_data_driven --configs train_data --set active_loss_names="('data', 'momentum', 'continuity', 'bc_walls', 'bc_lid', 'bc_pressure')" --set dataset_size=128 --set physics_on_data=False --set save_name=dataset_size_128 --set save_dir=./logs/ldc/cvit/physics_on_unlabeled_data/
python -m ldc.train_data_driven --configs train_data --set active_loss_names="('data', 'momentum', 'continuity', 'bc_walls', 'bc_lid', 'bc_pressure')" --set dataset_size=64 --set physics_on_data=False --set save_name=dataset_size_64 --set save_dir=./logs/ldc/cvit/physics_on_unlabeled_data/
python -m ldc.train_data_driven --configs train_data --set active_loss_names="('data', 'momentum', 'continuity', 'bc_walls', 'bc_lid', 'bc_pressure')" --set dataset_size=32 --set physics_on_data=False --set save_name=dataset_size_32 --set save_dir=./logs/ldc/cvit/physics_on_unlabeled_data/
``` 

Train with physics on labeled data:

```bash
python -m ldc.train_data_driven --configs train_data --set active_loss_names="('data', 'momentum', 'continuity', 'bc_walls', 'bc_lid', 'bc_pressure')" --set dataset_size=256 --set physics_on_data=True --set save_name=dataset_size_256 --set save_dir=./logs/ldc/cvit/physics_on_labeled_data/
python -m ldc.train_data_driven --configs train_data --set active_loss_names="('data', 'momentum', 'continuity', 'bc_walls', 'bc_lid', 'bc_pressure')" --set dataset_size=128 --set physics_on_data=True --set save_name=dataset_size_128 --set save_dir=./logs/ldc/cvit/physics_on_labeled_data/
python -m ldc.train_data_driven --configs train_data --set active_loss_names="('data', 'momentum', 'continuity', 'bc_walls', 'bc_lid', 'bc_pressure')" --set dataset_size=64 --set physics_on_data=True --set save_name=dataset_size_64 --set save_dir=./logs/ldc/cvit/physics_on_labeled_data/
python -m ldc.train_data_driven --configs train_data --set active_loss_names="('data', 'momentum', 'continuity', 'bc_walls', 'bc_lid', 'bc_pressure')" --set dataset_size=32 --set physics_on_data=True --set save_name=dataset_size_32 --set save_dir=./logs/ldc/cvit/physics_on_labeled_data/
```
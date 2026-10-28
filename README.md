# Lab2 - EEG Classification

EEGNet implemented in PyTorch for the **BCI Competition III, dataset IIIb** (S4b and X11b). Includes an optional **DeepConvNet** bonus experiment.

## Dataset

The four EEG `.npz` files from the teaching assistant are already inside `data/` in the downloadable ZIP. For a GitHub clone, place the four TA-provided files in `data/` yourself.

## Experiments

- EEGNet baseline: ELU (alpha=1.0), dropout=0.25
- EEGNet variants: ReLU, LeakyReLU, ELU alpha=0.5, dropout=0.50
- DeepConvNet: bonus model

The network is implemented manually, without pretrained weights. The EEGNet separable block uses depthwise temporal + pointwise convolution.

## How to run

Install a CUDA-enabled PyTorch compatible with your GPU first, then:

```bash
python -m pip install -r requirements.txt
python main.py --data-dir data --epochs 100 --batch-size 64
```

For a quick test of all functions (not an official result):

```bash
python main.py --smoke-test --epochs 1 --experiments baseline deepconvnet
```

To run only EEGNet baseline:

```bash
python main.py --experiments baseline --epochs 100 --batch-size 64
```

`results/` contains a CSV and chart for each experiment, a final comparison chart, model checkpoints, and JSON metrics. Models are selected by **validation accuracy**; epoch-wise test curves are reported for assignment diagnostics only, never used to select a checkpoint.

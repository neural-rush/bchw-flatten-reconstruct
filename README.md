# BCHW <-> B x (CHW) Tensor Flattening from Scratch

AI Accelerator Design - Assignment 3, IIT Tirupati.

Flattens a `B x C x H x W` tensor to `B x (CHW)` and reconstructs it, using only
loops and index arithmetic (no `reshape/view/flatten/ravel` in the algorithm),
then measures reconstruction error (Emax and MAE).

## Mapping

    flatten:      j = c*(H*W) + h*W + w
    reconstruct:  c = j // (H*W);  r = j % (H*W);  h = r // W;  w = r % W

## Structure

    .
    ├── README.md
    ├── requirements.txt
    ├── .gitignore
    ├── run_experiments.py        # main entry point: runs all cases, writes results/
    ├── src/
    │   ├── flatten.py            # core flatten / reconstruct (from scratch)
    │   ├── metrics.py            # Emax and MAE
    │   ├── data.py               # MNIST / CIFAR-10 loading, synthetic feature maps
    │   └── reference.py          # reshape-based check, REFERENCE ONLY
    ├── tests/
    │   └── test_roundtrip.py     # pytest round-trip and index tests
    ├── results/
    │   ├── results.csv
    │   └── results.md            # table used in the report
    └── report/
        └── report.md             # short report (export to PDF)

## Run

    pip install -r requirements.txt
    python run_experiments.py     # downloads MNIST/CIFAR-10 on first run
    python -m pytest -q

If `torchvision` is not installed, the script falls back to random tensors of
MNIST/CIFAR shape, so install it to get real-dataset results.

## Rules compliance

The only use of library `reshape` is `src/reference.py`, an optional independent
check that is not part of the submitted algorithm.

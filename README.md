# BCHW to B x (CHW): Tensor Flattening from Scratch

**Course:** AI Accelerator Design, IIT Tirupati | **Assignment 3**
**Author:** `<your name>`, Roll No. `<your roll number>`

This project flattens a `B x C x H x W` (BCHW/NCHW) tensor into a `B x (CHW)` matrix,
reconstructs it, and measures the reconstruction error. Both directions are implemented
**from scratch** with loops and index arithmetic. No `reshape`, `view`, `flatten` or
`ravel` is used in the algorithm.

**Live web app:** `https://<your-username>.github.io/<repo-name>/`
**Report:** [`report/report.pdf`](report/report.pdf)

---

## Quick start for reviewers

Pick whichever is easiest. Nothing needs to be installed for the first two.

| I want to... | Do this |
|---|---|
| **See the results and play with the mapping** | Open the live web app link above. |
| **Use the web app offline** | Download or clone the repo and double-click `docs/index.html`. |
| **Run the Python code** | Follow [Running the code](#running-the-code) below (about 2 minutes). |

### Using the web app

The page has four parts, from top to bottom:

1. **The mapping:** the two formulas used for flattening and reconstruction.
2. **Experimental results:** summary cards, the full results table (B, C, H, W, Emax, MAE,
   match against the library reference, run time), and a chart of time vs channel count.
3. **Mapping explorer:** shows a small tensor next to its flattened row, colour-coded by
   channel.
   - Change **C, H, W** (up to 6 / 6 / 8) to resize it.
   - **Hover** any cell to highlight the same element in the original grid and in the flat row.
   - Type **c, h, w** to see the flat index `j` with the arithmetic worked out, or type **j**
     to see the inverse. The defaults (C=2, H=2, W=3, element (1,1,2) to j=11) match the
     manual example in the report.
4. **Live round-trip test:** choose B, C (1 to 500), H and W and press **Run**. The browser
   generates a random tensor, flattens it, reconstructs it, and reports Emax, MAE and the
   time taken. The result should always be an exact reconstruction (error 0). Sizes are
   capped at 4,000,000 elements.

---

## The algorithm

Row-major layout, with `W` varying fastest:

```
flatten:      j = c*(H*W) + h*W + w

reconstruct:  c = j // (H*W)
              r = j %  (H*W)
              h = r // W
              w = r %  W
```

**Manual example** (B=1, C=2, H=2, W=3, so CHW = 12 and H*W = 6):

| Element | Calculation | Flat index j |
|---|---|---|
| I[0,0,0,0] | 0*6 + 0*3 + 0 | 0 |
| I[0,0,0,2] | 0*6 + 0*3 + 2 | 2 |
| I[0,0,1,0] | 0*6 + 1*3 + 0 | 3 |
| I[0,1,0,0] | 1*6 + 0*3 + 0 | 6 |
| I[0,1,1,2] | 1*6 + 1*3 + 2 | 11 |

Reverse check for j = 11: c = 11//6 = 1, r = 11%6 = 5, h = 5//3 = 1, w = 5%3 = 2.

The full pseudocode and discussion are in the [report](report/report.pdf).

---

## Running the code

**Requirements:** Python 3.9 or newer.

```bash
# 1. Get the code
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>

# 2. (Recommended) create a virtual environment
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run all experiments
python run_experiments.py

# 5. Run the tests
python -m pytest -q
```

**Notes**

- The first run downloads MNIST and CIFAR-10 into `data/` (about 170 MB, a few minutes).
- `torch` and `torchvision` are only needed to load MNIST/CIFAR. For a smaller download on a
  machine without a GPU:
  ```bash
  pip install numpy pytest
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
  ```
- If `torchvision` is not installed, the script still runs. It prints a notice and uses
  random tensors of MNIST/CIFAR shape instead, and the table rows are labelled
  `MNIST-like` / `CIFAR-like`.

### What you should see

The script first prints the manual example, then a results table with one row per
configuration:

```
Dataset/Input             B    C   H   W      Emax       MAE  ref ok  time(s)
MNIST / sample            2    1  28  28   0.0e+00   0.0e+00    True     0.00
CIFAR / sample            2    3  32  32   0.0e+00   0.0e+00    True     0.01
Synthetic fmap1           2    8  32  32   0.0e+00   0.0e+00    True     0.02
...
Synthetic fmap7           2  500  32  32   0.0e+00   0.0e+00    True     1.00
```

`Emax` and `MAE` are zero in every row because values are only copied, never computed on.
`ref ok = True` means the result matched an independent library `reshape` (used for
verification only). The tests should end with `6 passed`.

Outputs written:

- `results/results.csv` and `results/results.md` (the results table)
- `docs/results.js` (the data behind the web app)

---

## Experiments

B = 2 for all runs. MNIST uses 28x28, CIFAR-10 uses 32x32. Synthetic feature maps are
random normal tensors (standing in for DNN intermediate activations) with H = W = 32 and
C in {8, 16, 32, 64, 128, 256, 500}. Exact numbers are in `results/results.md` and in the
web app.

---

## Project structure

```
.
├── README.md
├── requirements.txt
├── .gitignore
├── run_experiments.py        # main entry point: runs all cases, writes results/ and docs/
├── src/
│   ├── flatten.py            # core flatten / reconstruct (from scratch)
│   ├── metrics.py            # Emax and MAE (explicit loops)
│   ├── data.py               # MNIST / CIFAR-10 loading, synthetic feature maps
│   └── reference.py          # reshape-based cross-check, REFERENCE ONLY
├── tests/
│   └── test_roundtrip.py     # round-trip and index tests (pytest)
├── docs/
│   ├── index.html            # web app (served by GitHub Pages)
│   └── results.js            # generated by run_experiments.py
├── results/
│   ├── results.csv
│   └── results.md
└── report/
    ├── report.md
    └── report.pdf
```

Where to look in the code: the core algorithm is in [`src/flatten.py`](src/flatten.py)
(`flatten_bchw`, `reconstruct_bchw`, and the index functions `flat_index` / `unflat_index`).

---

## Assignment rules compliance

- Flattening and reconstruction use loops and index arithmetic only (`src/flatten.py`).
- NumPy and torchvision are used only for loading data, generating tensors and storing
  elements (`src/data.py`).
- Library `reshape` appears in exactly one place, `src/reference.py`, as an optional
  independent check that is not part of the submitted algorithm.
- Error metrics (Emax, MAE) are computed with explicit loops (`src/metrics.py`).

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `ModuleNotFoundError: src` | Run commands from the project root folder. |
| `pip install` is very slow | Use the CPU-only torch command shown above. |
| Table says `MNIST-like` | torchvision is missing; install it and rerun. |
| Web app table is empty when opened locally | Run `python run_experiments.py` once to create `docs/results.js`. |
| Web app link gives a 404 | Repo Settings, Pages: deploy from branch `main`, folder `/docs`. |
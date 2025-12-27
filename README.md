# mace_batch_inference

This is an example of how a typical architecture for machine learning batch inference code would look like. Currently for demonstration purposes, the `uma_parallel.py` is a workable “single-file pipeline” script (POSCAR parsing → layer grouping → permutation generation → multiprocessing fan-out → GPU inference → screening → DB write). No modularization was implemented for simplicity sake. Uses UMA-S-1p1 (MD5 checksum = 36a2f071350be0ee4c15e7ebdd16dde1) as the pre-trained foundation model without any fine-tuning conducted (https://huggingface.co/facebook/UMA). Added also `analyze_uma_db.py` to showscase how to extract and analyze relevant information from the stored *.db files.

## Installation

### 1. Create and activate an environment
```bash
conda create -n ADD_YOUR_OWN_ENV_NAME python=3.10
conda activate ADD_YOUR_OWN_ENV_NAME
```
### 2. Clone repo
```bash
git clone [https://github.com/<your-org>/<your-repo>.git](https://github.com/pandabear245/uma_batch_inference/)
cd PATH_TO_GITHUB_REPO

```

### 3. Add execute permission to the folder where github repo is located in
`chmod +x uma_parallel.py`
`chmod +x analyze_uma_db.py`

### 4. Run uma_parallel.py 
`cd PATH_TO_POSCAR`
`python3 uma_parallel.py POSCAR --workers 16 --batch-size 512 --delta-e 0.5 --keep-frac 0.002 --energy-tol 1e-5 --db-path database/structures_uma_parallel.db`

### 5. Run analyze_uma_db.py
`cd database`
`python3 analyze_uma_db.py`

#!/bin/bash
# setup_universal.sh

echo "=== Universal Compute Efficiency Setup ==="

# 1. Python packages (all platforms)
echo "Installing high-performance dependencies..."
python3 -m pip install polars pyarrow numba threadpoolctl psutil joblib duckdb

# 2. LightGBM with OpenMP (Optional optimization)
# echo "Reinstalling LightGBM for OpenMP support..."
# python3 -m pip uninstall -y lightgbm
# python3 -m pip install lightgbm --no-binary :all:

# 3. Verify installations
echo "Verifying installations..."
python3 -c "
try:
    import polars, numba, threadpoolctl, duckdb, joblib
    print('Polars:', polars.__version__)
    print('Numba:', numba.__version__)
    print('DuckDB:', duckdb.__version__)
    print('Joblib:', joblib.__version__)
    print('Dependencies OK')
except ImportError as e:
    print('Missing dependency:', e)
    exit(1)
"

# 4. Run benchmark
echo "Running efficiency benchmarks..."
python3 -c "
import time
import polars as pl
import numpy as np
import pandas as pd
from pathlib import Path

# Benchmark 1: Polars vs pandas groupby
print('\n--- Benchmark 1: Polars GroupBy ---')
n_rows = 1_000_000
df_pd = pd.DataFrame({'a': np.random.randint(0, 1000, n_rows), 'b': np.random.random(n_rows)})
df_pl = pl.from_pandas(df_pd)

start = time.time()
res_pd = df_pd.groupby('a')['b'].mean()
print(f'Pandas 1M mean: {time.time()-start:.4f}s')

start = time.time()
res_pl = df_pl.group_by('a').agg(pl.col('b').mean())
print(f'Polars 1M mean: {time.time()-start:.4f}s')

# Benchmark 2: Numba AUC
print('\n--- Benchmark 2: Numba JIT Metrics ---')
from plasmid_priority.validation.fast_metrics import fast_auc
y = np.random.randint(0, 2, 100000)
s = np.random.random(100000)

# Warmup
fast_auc(y, s)

start = time.time()
for _ in range(100):
    fast_auc(y, s)
print(f'Numba 100x AUC: {time.time()-start:.4f}s')

from sklearn.metrics import roc_auc_score
start = time.time()
for _ in range(100):
    roc_auc_score(y, s)
print(f'Sklearn 100x AUC: {time.time()-start:.4f}s')

print('\n--- Efficiency Benchmarks Complete ---')
"

echo "=== Universal Compute Efficiency Setup Complete ==="

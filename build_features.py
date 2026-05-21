#!/usr/bin/env python3
"""One-time full indicator computation for all stocks. NEVER run incrementally."""
import os, pandas as pd, numpy as np, time
from indicators import calc_all_indicators

FEAT_DIR = 'ML_optimization/features'

def compute_one(f):
    path = f'{FEAT_DIR}/{f}'
    try:
        df = pd.read_parquet(path)
        if 'date' in df.columns:
            df = df.set_index('date').sort_index()
        if 'J' in df.columns and len(df) > 100:
            return (f, 0, 'skip')
        df = calc_all_indicators(df, board_type='main')
        df.to_parquet(path)
        return (f, len(df), 'ok')
    except Exception as e:
        return (f, 0, str(e))

files = sorted([f for f in os.listdir(FEAT_DIR) if f.endswith('.parquet')])
print(f'Total files: {len(files)}')
done = 0; skipped = 0; errors = 0
t0 = time.time()
for i, f in enumerate(files):
    name, n, status = compute_one(f)
    if status == 'ok': done += 1
    elif status == 'skip': skipped += 1
    else: errors += 1
    if (i+1) % 100 == 0:
        elapsed = time.time() - t0
        rate = (i+1) / elapsed
        remaining = (len(files) - i - 1) / rate
        print(f'  {i+1}/{len(files)} done={done} skip={skipped} err={errors} ETA={remaining/60:.0f}m')

print(f'Done: {done} computed, {skipped} skipped, {errors} errors in {time.time()-t0:.0f}s')

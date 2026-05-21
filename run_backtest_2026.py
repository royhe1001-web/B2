#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Spring B2 朴素版回测 -- P50市值过滤, 自定义入场+离场规则
"""
import sys, os, time, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import pandas as pd
import numpy as np
import logging
logging.basicConfig(level=logging.WARNING, format='%(asctime)s [%(levelname)s] %(message)s')

from ML_optimization.phase2c_oamv_grid_search import OAMVSimEngine
import ML_optimization.phase2c_oamv_grid_search as p2c_mod
from ML_optimization.phase2c_bull_grid_search import _get_stock_files_mainboard, preload_stock_data
from ML_optimization.mktcap_utils import (
    build_mktcap_lookup, compute_mktcap_percentiles, get_pct_universe,
)
from oamv import fetch_market_data, calc_oamv, generate_signals

BASE = os.path.dirname(os.path.abspath(__file__))

# 朴素版硬编码参数
BEST_B2 = {
    'j_prev_max': 20, 'gain_min': 0.04, 'j_today_max': 65,
    'shadow_max': 0.035, 'j_today_loose': 70, 'shadow_loose': 0.045,
    'prior_strong_ret': 0.20,
    'weight_gain_2x_thresh': 0.0842, 'weight_gain_2x': 1.479,
    'weight_gap_thresh': 0.02, 'weight_gap_up': 1.5,
    'weight_shrink': 1.171, 'weight_deep_shrink_ratio': 0.80,
    'weight_deep_shrink': 1.3, 'weight_pullback_thresh': -0.05,
    'weight_pullback': 1.2, 'weight_shadow_discount_thresh': 0.015,
    'weight_shadow_discount': 0.7, 'weight_strong_discount': 0.8,
    'weight_brick_resonance': 1.5,
    'sector_momentum_enabled': False,
    'dynamic_thresholds': False, 'trend_filter': False,
}

BEST_OAMV = {
    'mktcap_lower_pct': 50, 'mktcap_upper_pct': 96,
}


def load_all(test_start, test_end):
    print('=' * 70)
    print(f'  Spring B2 -- P50-P96, 5-exit rules')
    print(f'  {test_start.date()} ~ {test_end.date()}')
    print('=' * 70)

    p2c_mod.SIM_START = test_start
    p2c_mod.SIM_END = test_end

    t0 = time.time()
    mktcap_lookup = build_mktcap_lookup()
    percentiles = compute_mktcap_percentiles()
    print(f'  Market cap: {len(mktcap_lookup)} entries ({time.time()-t0:.0f}s)')

    ym_start = test_start.strftime('%Y-%m')
    ym_end = test_end.strftime('%Y-%m')
    eligible_codes = get_pct_universe(mktcap_lookup, percentiles, ym_start,
                                      ym_end=ym_end, lower_pct=50, upper_pct=96)
    print(f'  P50-P96: {len(eligible_codes)} stocks')

    t0 = time.time()
    oamv_df = fetch_market_data(start='20180101')
    oamv_df = calc_oamv(oamv_df)
    oamv_df = generate_signals(oamv_df)
    print(f'  OAMV: {len(oamv_df)} days ({time.time()-t0:.0f}s)')

    t0 = time.time()
    stock_data = preload_stock_data(eligible_codes)
    print(f'  K-line: {len(stock_data)} stocks ({time.time()-t0:.0f}s)')

    return stock_data, oamv_df, mktcap_lookup, percentiles


def main():
    parser = argparse.ArgumentParser(description='Spring B2 朴素版回测')
    parser.add_argument('--start', type=str, default=None)
    parser.add_argument('--end', type=str, default=None)
    args = parser.parse_args()

    today = pd.Timestamp.now()
    test_start = pd.Timestamp(args.start) if args.start else pd.Timestamp(f'{today.year}-01-01')
    test_end = pd.Timestamp(args.end) if args.end else today

    stock_data, oamv_df, mktcap_lookup, percentiles = load_all(test_start, test_end)

    last_oamv = pd.Timestamp(oamv_df['date'].max())
    if test_end > last_oamv:
        test_end = last_oamv
        p2c_mod.SIM_END = test_end

    print(f'\n{"="*70}')
    print(f'  B2 Spring: P50-P96, 5-exit rules')
    print(f'{"="*70}')

    t0 = time.time()
    engine = OAMVSimEngine(stock_data, BEST_B2, oamv_df, BEST_OAMV,
                           mktcap_lookup=mktcap_lookup,
                           percentiles=percentiles)
    m = engine.run()
    elapsed = time.time() - t0

    print(f'\n  Return={m["total_return_pct"]:+.1f}%  '
          f'Sharpe={m["sharpe"]:.3f}  WR={m["win_rate"]:.1%}  '
          f'Trades={m["n_trades"]}  MaxDD={m["max_dd"]:.1%}  '
          f'({elapsed:.0f}s)')

    # 导出期末持仓供实时卖出检测
    import json as _json
    sig_dir = os.path.join(BASE, 'output', 'signals')
    os.makedirs(sig_dir, exist_ok=True)
    positions = []
    for code, pos in engine.positions.items():
        df = stock_data.get(code)
        if df is not None and len(df) > 0:
            last_row = df.iloc[-1]
            positions.append({
                'symbol': code,
                'entry_price': round(pos.buy_price, 2),
                'entry_low': round(pos.entry_low, 2),
                'entry_date': str(pos.buy_date.date()),
                'signal_close': round(pos.signal_close, 2),
                'signal_weight': round(pos.signal_weight, 2),
                'half_sold': pos.half_sold,
                'shares': pos.shares,
                'last_close': round(float(last_row['close']), 2),
                'return_pct': round((float(last_row['close']) / pos.buy_price - 1) * 100, 1),
            })
    with open(os.path.join(sig_dir, 'current_positions.json'), 'w') as f:
        _json.dump(positions, f, ensure_ascii=False, indent=2)
    print(f'  EOD positions: {len(positions)} -> output/signals/current_positions.json')
    if positions:
        for p in positions:
            sym = p['symbol']; ep = p['entry_price']; sw = p['signal_weight']; rp = p['return_pct']
            print(f'    {sym:>8s} entry={ep:.2f} weight={sw:.1f} ret={rp:+.1f}%')
    print()


if __name__ == '__main__':
    main()

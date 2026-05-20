#!/usr/bin/env python3
"""B2 Spring TPE 多窗口网格搜索 — 权重 + 止损联合优化"""
import sys, os, time, json, numpy as np, pandas as pd
from pathlib import Path
from datetime import datetime
import logging; logging.basicConfig(level=logging.WARNING)

ROOT = Path(__file__).parent
os.chdir(str(ROOT)); sys.path.insert(0, str(ROOT))

try:
    import optuna
    from optuna.samplers import TPESampler
    from optuna.pruners import MedianPruner
except ImportError:
    print("pip install optuna"); sys.exit(1)

from config import *
import ML_optimization.phase2c_oamv_grid_search as engine_mod
from ML_optimization.phase2c_bull_grid_search import preload_stock_data, _get_stock_files_mainboard
from ML_optimization.mktcap_utils import build_mktcap_lookup, compute_mktcap_percentiles, get_pct_universe
from oamv import fetch_market_data, calc_oamv, generate_signals

# ── 数据缓存 ──
_cache = {}

def get_cached_data():
    if not _cache:
        print("  [缓存] 加载数据...")
        _cache['mktcap_lookup'] = build_mktcap_lookup()
        _cache['percentiles'] = compute_mktcap_percentiles()
        oamv_df = fetch_market_data(start='20100101')
        oamv_df = calc_oamv(oamv_df)
        _cache['oamv_df'] = generate_signals(oamv_df)
        # 单窗口全量主板股票
        all_codes = {os.path.basename(f).replace('.parquet', '') for f in _get_stock_files_mainboard()}
        _cache['stock_data'] = preload_stock_data(all_codes)
        print(f"  [缓存] {len(_cache['stock_data'])} stocks ready (全量)")
    return _cache


def suggest_params(trial):
    p = {}
    for name, (lo, hi) in TPE_PARAM_SPACE.items():
        if any(k in name for k in ('pos_intercept', 'pos_slope', 'qvix_offset',
                                     'tail_qvix_min', 'recovery_qvix_peak', 't1_threshold', 'max_signal_days', 'vol_check_days', 'crash_min')):
            p[name] = trial.suggest_int(name, int(lo), int(hi))
        elif 'tail_ratio' in name:
            p[name] = trial.suggest_float(name, lo, hi, step=0.05)
        else:
            p[name] = trial.suggest_float(name, lo, hi)
    return p


def apply_params(trial_params):
    """Distribute trial params to B2 and stop configs."""
    # B2 params: base BEST_B2 + trial overrides
    b2 = BEST_B2.copy()
    for k in ('weight_pullback_thresh','rotate_base','sig_expire_thresh','max_signal_days',
              'vol_check_days','vol_spike_thresh','vol_contraction_ratio',
              'weight_gain_2x_thresh','weight_gain_2x','weight_shrink','j_prev_max','gain_min'):
        if k in trial_params:
            b2[k] = trial_params[k]

    # Stop config: FROZEN_STOP_DEFAULTS + trial overrides
    stop_cfg = dict(FROZEN_STOP_DEFAULTS)
    for tk in ('trailing_activate','trailing_pct','take_profit_partial',
               'take_profit_partial_frac','breakeven_thresh','peak_dd_stop'):
        if tk in trial_params:
            stop_cfg[tk] = trial_params[tk]
    if 'hard_stop_aggressive' in trial_params:
        stop_cfg['hard_stop'] = {'aggressive': trial_params['hard_stop_aggressive'],
                                  'normal': -0.10, 'defensive': -0.08}

    return b2, stop_cfg


def run_one_window(params, start, end):
    """Run backtest on one window, return (strategy_return%, max_dd%, trading_days)."""
    b2, stop_cfg = apply_params(params)
    data = get_cached_data()

    engine = engine_mod.OAMVSimEngine(
        data['stock_data'], b2, data['oamv_df'], {},
        mktcap_lookup=data['mktcap_lookup'], percentiles=data['percentiles'],
        stop_config=stop_cfg)
    engine_mod.SIM_START = pd.Timestamp(start)
    engine_mod.SIM_END = pd.Timestamp(end)
    result = engine.run()
    total_ret = result['total_return_pct']
    max_dd = result['max_dd']
    n_days = len(data['oamv_df'][(data['oamv_df']['date']>=start)&(data['oamv_df']['date']<=end)])
    return total_ret, max_dd, n_days


def objective(trial):
    trial_num = trial.number
    params = suggest_params(trial)
    print(f"  [Trial {trial_num}] params: {json.dumps(params, default=str)}", flush=True)
    try:
        total_ret, max_dd, n_days = run_one_window(params, *TRAIN_WINDOWS[0][1:])
        if n_days <= 1 or total_ret <= -100:
            score = -999.0
        else:
            score = total_ret
        print(f"  [Trial {trial_num}] ret={score:.1f}% dd={max_dd:.1%}", flush=True)
    except Exception as e:
        print(f"    [Trial {trial_num}] CRASH: {e}", flush=True)
        score = -999.0
    return score


def validate_best(params, windows, seeds=(42, 123, 456)):
    results = {}
    for wname, wstart, wend in windows:
        runs = []
        for seed in seeds:
            np.random.seed(seed)
            total_ret, max_dd, n_days = run_one_window(params, wstart, wend)
            cagr = (1 + total_ret/100) ** (252.0 / max(n_days, 1)) - 1
            score = cagr * (1 - min(max_dd, 0.99)) ** 2
            runs.append({'seed': seed, 'total_ret': total_ret, 'max_dd': max_dd, 'score': score})
        results[wname] = {
            'ret_mean': np.mean([r['total_ret'] for r in runs]),
            'dd_mean': np.mean([r['max_dd'] for r in runs]),
            'score_mean': np.mean([r['score'] for r in runs]),
            'runs': runs,
        }
    return results


def main():
    import argparse as ap
    ap = ap.ArgumentParser(description='B2 Spring TPE 多窗口优化')
    ap.add_argument('--trials', type=int, default=300)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--resume', action='store_true', help='续跑已有study')
    args = ap.parse_args()

    print("=" * 70)
    print(f"  B2 Spring TPE 多窗口优化 — {args.trials} trials")
    for n, s, e in TRAIN_WINDOWS:
        print(f"    {n}: {s} ~ {e}")
    print("=" * 70)

    get_cached_data()

    db_path = ROOT / 'output' / 'spring_tpe_v9.db'
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if not args.resume and db_path.exists():
        db_path.unlink()

    t0 = time.time()
    study = optuna.create_study(
        study_name="spring_b2_v6", storage=f"sqlite:///{db_path}",
        sampler=TPESampler(seed=args.seed),
        pruner=MedianPruner(n_startup_trials=10, n_warmup_steps=2),
        direction="maximize",
        load_if_exists=args.resume,
    )

    # 用callback保存每轮结果 (确保中途中断可恢复)
    def save_callback(study, trial):
        if trial.state == optuna.trial.TrialState.COMPLETE:
            best = study.best_params
            out = {'best_params': best, 'best_score': study.best_value,
                   'n_trials': len(study.trials)}
            out_path = ROOT / 'output' / 'spring_tpe_results.json'
            with open(out_path, 'w', encoding='utf-8') as f:
                json.dump(out, f, indent=2, ensure_ascii=False, default=str)

    done = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    remaining = max(args.trials - done, 0)
    print(f"  已完成: {done} trials, 剩余: {remaining}")
    if remaining <= 0:
        print("  已全部完成!")
        return

    study.optimize(objective, n_trials=remaining, show_progress_bar=True,
                   callbacks=[save_callback], n_jobs=1)
    elapsed = time.time() - t0

    best = study.best_params
    print(f"\n  搜索完成 ({elapsed:.0f}s)")
    print(f"  Best score: {study.best_value:.4f}")
    for k, v in sorted(best.items()):
        print(f"    {k:30s} = {v}")

    # Validate
    print(f"\n  [验证] Val + Test + Bear...")
    today = datetime.now().strftime('%Y-%m-%d')
    val_results = validate_best(best, [VAL_WINDOW, ('Test_2026', '2026-01-01', today),
                                        BEAR_STRESS])

    for wname, wr in val_results.items():
        print(f"  {wname}: ret={wr['ret_mean']:.1f}% dd={wr['dd_mean']:.1%} score={wr['score_mean']:.4f}")

    out = {'best_params': best, 'best_score': study.best_value, 'validation': val_results}
    out_path = ROOT / 'output' / 'spring_tpe_results.json'
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\n  结果: {out_path}")


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""B2 收盘前卖出检测 — 14:55 拉实时价, 与回测引擎 check_stops 完全一致"""
import sys, os, json, requests, time
from pathlib import Path
from datetime import datetime
import pandas as pd, numpy as np

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
os.chdir(str(ROOT))

from config import FROZEN_STOP_DEFAULTS, ENGINE_CONFIG

# ── 实时行情 ──
def fetch_sina_realtime(symbols: list) -> pd.DataFrame:
    session = requests.Session(); session.trust_env = False
    all_data = []
    for i in range(0, len(symbols), 200):
        batch = symbols[i:i+200]
        sina_codes = [f'sh{c}' if c.startswith(('60','68','9')) else f'sz{c}' for c in batch]
        url = 'https://hq.sinajs.cn/list=' + ','.join(sina_codes)
        r = session.get(url, headers={'Referer': 'https://finance.sina.com.cn'}, timeout=10)
        r.encoding = 'gbk'
        for line in r.text.strip().split('\n'):
            if '=' not in line: continue
            code_str, quote_str = line.split('=', 1)
            code = code_str.strip().split('_')[-1][2:]
            fields = quote_str.strip('";\n ').split(',')
            if len(fields) < 32: continue
            all_data.append({
                'symbol': code,
                'name': fields[0], 'open': float(fields[1]) if fields[1] else 0,
                'pre_close': float(fields[2]) if fields[2] else 0,
                'price': float(fields[3]) if fields[3] else 0,
                'high': float(fields[4]) if fields[4] else 0,
                'low': float(fields[5]) if fields[5] else 0,
            })
        time.sleep(0.1)
    df = pd.DataFrame(all_data)
    if not df.empty: df = df.set_index('symbol')
    return df

# ── 加载历史数据 ──
def load_recent_data(symbols: list, days=120):
    import pickle
    cache_file = ROOT / 'output' / 'close_sell_cache.pkl'
    cache = {}
    if cache_file.exists():
        try:
            with open(cache_file, 'rb') as f: cache = pickle.load(f)
        except: pass

    data_dir = Path('ML_optimization/features')
    result = {}
    for sym in symbols:
        if sym in cache:
            result[sym] = cache[sym]
            continue
        fp = data_dir / f'{sym}.parquet'
        if fp.exists():
            df = pd.read_parquet(fp)
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
                df = df.set_index('date').sort_index()
            result[sym] = df.tail(days)
        else:
            print(f'  {sym}: data not found')
    for sym, df in result.items():
        cache[sym] = df
    with open(cache_file, 'wb') as f: pickle.dump(cache, f)
    return result

# ── 卖出条件评估 (与回测引擎 check_stops 完全一致) ──
def check_exit(pos: dict, recent: pd.DataFrame, rt_price: float, regime: str = 'normal'):
    """
    与 OAMVSimEngine.check_stops() 保持完全一致:
      1. 60d弱势  2. 252d到期  3. take_profit_full
      4. take_profit_partial  5. peak_dd  6. trail
      7. candle  8. signal_close  9. yellow
    """
    entry_price = pos['entry_price']
    entry_low = pos.get('entry_low', entry_price)
    signal_close = pos.get('signal_close', entry_price)
    half_sold = pos.get('half_sold', False)
    entry_date = pd.Timestamp(pos['entry_date'])
    held_days = (pd.Timestamp.now().normalize() - entry_date).days
    cum_return = rt_price / entry_price - 1

    # 历史峰值 (从买入至今)
    if entry_date in recent.index:
        post_entry = recent[recent.index >= entry_date]
    else:
        post_entry = recent
    peak_price = float(post_entry['high'].max()) if len(post_entry) > 0 else rt_price

    # ── 参数来源: stop_config(FROZEN_STOP_DEFAULTS) → oamv_params(BEST_OAMV) → 默认 ──
    trail_activate = FROZEN_STOP_DEFAULTS.get('trailing_activate', 0.15)
    trail_pct      = FROZEN_STOP_DEFAULTS.get('trailing_pct', 0.08)
    tp_partial     = FROZEN_STOP_DEFAULTS.get('take_profit_partial', 0.30)
    tp_partial_frac = FROZEN_STOP_DEFAULTS.get('take_profit_partial_frac', 0.50)
    tp_full        = FROZEN_STOP_DEFAULTS.get('take_profit_full', 0.60)
    peak_dd_limit  = FROZEN_STOP_DEFAULTS.get('peak_dd_stop', -0.12)

    # OAMV 蜡烛缓冲 (默认 normal=0.97, 来自 run_backtest BEST_OAMV)
    candle_buf = 0.97
    grace_days = 2

    # 1. 60天弱势 (与引擎一致: >=60天且 <15%)
    if held_days >= 60 and cum_return < 0.15:
        return True, f'60d_weak(T+{held_days},{cum_return*100:+.0f}%)', 1.0

    # 2. 252天到期
    if held_days >= 252:
        return True, f'1yr_clear(T+{held_days})', 1.0

    # 3. 分批止盈全卖 (+60%)
    if cum_return >= tp_full:
        return True, f'take_full', 1.0

    # 4. 分批止盈卖半 (+30%, 未卖过半)
    if cum_return >= tp_partial and not half_sold:
        return True, f'take_half', tp_partial_frac

    # 5. 峰值回撤止损
    if peak_price > 0 and (rt_price / peak_price - 1) <= peak_dd_limit:
        return True, f'peak_dd', 1.0

    # 6. 移动止盈
    if (cum_return >= trail_activate and peak_price > entry_price
            and rt_price < peak_price * (1 - trail_pct)):
        return True, f'trail', 1.0

    # 7. 蜡烛止损 (收盘 < 入场最低 * buffer, 且持有>=免死期)
    candle_level = entry_low * candle_buf
    if rt_price < candle_level and held_days >= grace_days:
        return True, f'candle(T+{held_days})', 1.0

    # 8. 信号止损 (收盘 < 信号日收盘)
    if rt_price < signal_close:
        return True, f'sig_close(T+{held_days})', 1.0

    # 9. 黄线止损
    yellow_line = float(recent['yellow_line'].iloc[-1]) if 'yellow_line' in recent.columns else 0
    if yellow_line > 0 and rt_price < yellow_line:
        return True, f'yellow(T+{held_days})', 1.0

    return False, '', 0

# ── 主流程 ──
def main():
    print(f'{"="*55}')
    print(f'  B2 收盘前卖出检测 — {datetime.now().strftime("%Y-%m-%d %H:%M")}')
    print(f'{"="*55}')

    pos_file = ROOT / 'output' / 'signals' / 'current_positions.json'
    if not pos_file.exists():
        print('  无持仓文件, 请先运行 B2 回测生成持仓')
        return
    with open(pos_file) as f:
        positions = json.load(f)
    if not positions:
        print('  当前无持仓')
        return

    symbols = [p['symbol'] for p in positions]
    print(f'  持仓 {len(positions)} 只: {symbols}')

    print('  拉取实时行情...')
    rt = fetch_sina_realtime(symbols)
    print(f'  获取 {len(rt)} 只')

    print('  加载历史数据...')
    data = load_recent_data(symbols)

    sells, holds = [], []
    for p in positions:
        sym = p['symbol']
        if sym not in rt.index:
            print(f'  {sym}: 无实时行情, 跳过'); holds.append(p); continue
        if sym not in data:
            print(f'  {sym}: 无历史数据, 跳过'); holds.append(p); continue

        rt_price = rt.loc[sym, 'price']
        recent = data[sym]
        sell, reason, partial = check_exit(p, recent, rt_price)
        ret = (rt_price / p['entry_price'] - 1) * 100

        if sell:
            sells.append({**p, 'rt_price': rt_price, 'ret': ret, 'reason': reason, 'partial': partial})
        else:
            holds.append({**p, 'rt_price': rt_price, 'ret': ret})

    if sells:
        print(f'\n  卖出信号 ({len(sells)}只):')
        for s in sells:
            tag = '(卖半)' if s['partial'] < 1.0 else ''
            print(f'    {s["symbol"]:8s} 现价{s["rt_price"]:.2f} 盈亏{s["ret"]:+.1f}% | {s["reason"]} {tag}')
    else:
        print(f'\n  无卖出信号')

    print(f'\n  继续持有 ({len(holds)}只):')
    for h in holds:
        print(f'    {h["symbol"]:8s} 现价{h.get("rt_price",0):.2f} 盈亏{h.get("ret",0):+.1f}%')

    today = datetime.now().strftime('%Y%m%d')
    sig_dir = ROOT / 'output' / 'signals'
    sig_dir.mkdir(parents=True, exist_ok=True)
    signal_data = {
        'date': str(datetime.now().date()),
        'time': str(datetime.now().time()),
        'sells': [{k: str(v) if isinstance(v, pd.Timestamp) else v for k,v in s.items()} for s in sells],
        'holds': [{k: str(v) if isinstance(v, pd.Timestamp) else v for k,v in h.items()} for h in holds],
    }
    with open(sig_dir / f'{today}_close_sell.json', 'w') as f:
        json.dump(signal_data, f, ensure_ascii=False, indent=2, default=str)
    print(f'\n  信号已保存: {sig_dir / f"{today}_close_sell.json"}')

if __name__ == '__main__':
    main()

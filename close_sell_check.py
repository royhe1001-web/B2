#!/usr/bin/env python3
"""B2 收盘前卖出检测 — 与回测引擎完全一致 (5条离场 + rotate轮换扫描)"""
import sys, os, json, requests, time
from pathlib import Path
from datetime import datetime
import pandas as pd, numpy as np

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
os.chdir(str(ROOT))

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
            result[sym] = cache[sym]; continue
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

# ── 离场检测 (5条, 与引擎 check_stops 完全一致) ──
def check_exit(pos: dict, recent: pd.DataFrame, rt_price: float):
    """1.take_half(+10%) 2.white_line 3.yellow_line 4.candle 5.time_stop(4d)"""
    entry_price = pos['entry_price']
    entry_low = pos.get('entry_low', entry_price)
    half_sold = pos.get('half_sold', False)
    entry_date = pd.Timestamp(pos['entry_date'])
    held_days = (pd.Timestamp.now().normalize() - entry_date).days
    cum_return = rt_price / entry_price - 1

    if cum_return >= 0.10 and not half_sold:
        return True, f'take_half(+{cum_return*100:.0f}%)', 0.50
    white_line = float(recent['white_line'].iloc[-1]) if 'white_line' in recent.columns else 0
    if white_line > 0 and rt_price < white_line:
        return True, f'white_line(T+{held_days})', 1.0
    yellow_line = float(recent['yellow_line'].iloc[-1]) if 'yellow_line' in recent.columns else 0
    if yellow_line > 0 and rt_price < yellow_line:
        return True, f'yellow_line(T+{held_days})', 1.0
    if rt_price < entry_low:
        return True, f'candle(T+{held_days})', 1.0
    if held_days >= 4 and cum_return < 0:
        return True, f'time_stop(T+{held_days})', 1.0
    return False, '', 0

# ── 轮换检测 (与引擎 rotate_if_needed 完全一致) ──
def check_rotate(positions: list, recent_data: dict, rt_prices: dict,
                 b2_params: dict = None, oamv_params: dict = None):
    """扫描今日Spring信号, 判断是否触发轮换"""
    from strategy_spring import generate_spring_signals
    from ML_optimization.mktcap_utils import is_in_pct_range, build_mktcap_lookup, compute_mktcap_percentiles

    if not positions: return []

    # 生成今日信号
    today_sigs = []
    for p in positions:
        sym = p['symbol']
        if sym not in recent_data: continue
        df = recent_data[sym].copy()
        if len(df) < 60: continue
        b2p = dict(b2_params or {})
        try:
            sdf = generate_spring_signals(df, board_type='main', precomputed=True, params=b2p)
            last = sdf.index[-1]
            if sdf.loc[last, 'b2_entry_signal'] == 1:
                weight = float(sdf.loc[last, 'b2_position_weight'])
                today_sigs.append({'code': sym, 'weight': weight, 'close': float(sdf.loc[last, 'close'])})
        except: pass

    if not today_sigs: return []

    # 市值过滤
    try:
        mkt = build_mktcap_lookup()
        pct = compute_mktcap_percentiles()
        lo_pct = (oamv_params or {}).get('mktcap_lower_pct', 50)
        hi_pct = (oamv_params or {}).get('mktcap_upper_pct', 96)
        today_sigs = [s for s in today_sigs
                      if is_in_pct_range(s['code'], pd.Timestamp.now(), mkt, pct,
                                         lower_pct=lo_pct, upper_pct=hi_pct)]
    except: pass

    # 按权重排序
    today_sigs.sort(key=lambda x: -x['weight'])
    rotate_base = (b2_params or {}).get('rotate_base', 0.3)

    rotates = []
    for sig in today_sigs:
        for p in positions:
            if p['symbol'] == sig['code']: continue
            pos_weight = p.get('signal_weight', p.get('weight', 1.0))
            if sig['weight'] > pos_weight + rotate_base:
                # Check position return
                rt_px = rt_prices.get(p['symbol'], 0)
                if rt_px > 0:
                    pos_ret = rt_px / p['entry_price'] - 1
                    if pos_ret < 0.05:  # flat or losing position
                        rotates.append({
                            'sell_code': p['symbol'], 'buy_code': sig['code'],
                            'buy_weight': sig['weight'], 'sell_weight': pos_weight,
                            'reason': f'rotate: {sig["code"]}({sig["weight"]:.1f}) > {p["symbol"]}({pos_weight:.1f})+0.3'
                        })
    return rotates

# ── 主流程 ──
def main():
    print(f'{"="*55}')
    print(f'  B2 收盘前卖出检测 — {datetime.now().strftime("%Y-%m-%d %H:%M")}')
    print(f'{"="*55}')

    pos_file = ROOT / 'output' / 'signals' / 'current_positions.json'
    if not pos_file.exists():
        print('  无持仓文件')
        return
    with open(pos_file) as f:
        positions = json.load(f)
    if not positions:
        print('  当前无持仓')
        return

    symbols = [p['symbol'] for p in positions]
    print(f'  持仓 {len(positions)} 只: {symbols}')

    # 拉行情
    print('  拉取实时行情...')
    rt = fetch_sina_realtime(symbols)
    print(f'  获取 {len(rt)} 只')

    # 加载历史
    print('  加载历史数据...')
    data = load_recent_data(symbols)

    # 评估离场
    sells, holds = [], []
    for p in positions:
        sym = p['symbol']
        if sym not in rt.index or sym not in data:
            holds.append(p); continue
        rt_price = rt.loc[sym, 'price']
        recent = data[sym]
        sell, reason, partial = check_exit(p, recent, rt_price)
        ret = (rt_price / p['entry_price'] - 1) * 100
        if sell:
            sells.append({**p, 'rt_price': rt_price, 'ret': ret, 'reason': reason, 'partial': partial})
        else:
            holds.append({**p, 'rt_price': rt_price, 'ret': ret})

    # 轮换检测
    rotates = []
    try:
        from run_backtest_2026 import BEST_B2, BEST_OAMV
        rt_prices = {s: rt.loc[s, 'price'] for s in symbols if s in rt.index}
        rotates = check_rotate(holds if not sells else positions, data, rt_prices, BEST_B2, BEST_OAMV)
    except Exception as e:
        print(f'  轮换检测跳过: {e}')

    # 输出
    if sells:
        print(f'\n  卖出信号 ({len(sells)}只):')
        for s in sells:
            tag = '(卖半)' if s['partial'] < 1.0 else ''
            print(f'    {s["symbol"]:8s} 现价{s["rt_price"]:.2f} 盈亏{s["ret"]:+.1f}% | {s["reason"]} {tag}')

    if rotates:
        print(f'\n  轮换信号 ({len(rotates)}组):')
        for r in rotates:
            print(f'    卖{r["sell_code"]} → 买{r["buy_code"]} | {r["reason"]}')

    if not sells and not rotates:
        print(f'\n  无卖出/轮换信号')

    print(f'\n  继续持有 ({len(holds)}只):')
    for h in holds[:10]:
        print(f'    {h["symbol"]:8s} 现价{h.get("rt_price",0):.2f} 盈亏{h.get("ret",0):+.1f}%')

    # 保存
    today = datetime.now().strftime('%Y%m%d')
    sig_dir = ROOT / 'output' / 'signals'
    sig_dir.mkdir(parents=True, exist_ok=True)
    signal_data = {
        'date': str(datetime.now().date()), 'time': str(datetime.now().time()),
        'sells': [{k: str(v) if isinstance(v, pd.Timestamp) else v for k,v in s.items()} for s in sells],
        'rotates': rotates,
        'holds': [{k: str(v) if isinstance(v, pd.Timestamp) else v for k,v in h.items()} for h in holds],
    }
    with open(sig_dir / f'{today}_close_sell.json', 'w') as f:
        json.dump(signal_data, f, ensure_ascii=False, indent=2, default=str)
    print(f'\n  信号已保存: output/signals/{today}_close_sell.json')

if __name__ == '__main__':
    main()

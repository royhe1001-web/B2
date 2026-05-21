#!/usr/bin/env python3
"""B2 Spring 每日运营报告"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding='utf-8')
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd, numpy as np
from ML_optimization.phase2c_oamv_grid_search import OAMVSimEngine
import ML_optimization.phase2c_oamv_grid_search as p2c_mod
from ML_optimization.phase2c_bull_grid_search import preload_stock_data
from ML_optimization.mktcap_utils import build_mktcap_lookup, compute_mktcap_percentiles, get_pct_universe
from oamv import fetch_market_data, calc_oamv, generate_signals
from run_backtest_2026 import BEST_B2, BEST_OAMV

today = pd.Timestamp.now().normalize()
p2c_mod.SIM_START = pd.Timestamp(f'{today.year}-01-01')
p2c_mod.SIM_END = today

# Clear caches
for f in ['output/mktcap_lookup.pkl', 'output/mktcap_percentiles.pkl']:
    if os.path.exists(f): os.remove(f)
for f in os.listdir('output'):
    if f.startswith('signal_cache_') and f.endswith('.pkl'):
        os.remove(os.path.join('output', f))

mkt = build_mktcap_lookup(); pct = compute_mktcap_percentiles()
eli = get_pct_universe(mkt, pct, f'{today.year}-01', ym_end=today.strftime('%Y-%m'), lower_pct=50, upper_pct=96)
oamv = fetch_market_data(start='20180101'); oamv = calc_oamv(oamv); oamv = generate_signals(oamv)
sd = preload_stock_data(eli)
e = OAMVSimEngine(sd, BEST_B2, oamv, BEST_OAMV, mktcap_lookup=mkt, percentiles=pct)
r = e.run()

dv = pd.DataFrame(e.daily_values)
pos = json.load(open('output/signals/current_positions.json'))

today_val = dv['value'].iloc[-1]; yesterday_val = dv['value'].iloc[-2]
today_pnl = today_val - yesterday_val
initial = 100000
total_ret = (today_val/initial - 1) * 100
today_ret = (today_val/yesterday_val - 1) * 100

print('=' * 55)
print(f'  B2 Spring 每日运营报告 — {str(dv["date"].iloc[-1])[:10]}')
print('=' * 55)
print(f'\n【策略总览】')
print(f'  策略总资本:   {today_val:,.0f}')
print(f'  策略仓位:     {len(pos)} 只')
print(f'  策略今日收益: {today_ret:+.2f}% ({today_pnl:+,.0f})')
print(f'  策略总收益:   {total_ret:+.1f}% ({today_val - initial:+,.0f})')

print(f'\n【持仓明细】')
total_pos_value = 0
for p in pos:
    code = p['symbol']; entry_date = p['entry_date']; entry_px = p['entry_price']
    shares = p['shares']; last_px = p['last_close']
    pos_value = shares * last_px; total_pos_value += pos_value
    held_days = (today - pd.Timestamp(entry_date)).days
    stock_df = sd.get(code)
    stock_today_pnl = 0
    if stock_df is not None and len(stock_df) >= 2:
        stock_today_pnl = (last_px - float(stock_df.iloc[-2]['close'])) * shares
    print(f'  {code}:')
    print(f'    持仓天数: {held_days}天 | 股数: {shares}股')
    print(f'    入场价: {entry_px:.2f} | 现价: {last_px:.2f}')
    print(f'    个股持仓市值: {pos_value:,.0f}')
    print(f'    个股今日盈利: {stock_today_pnl:+,.0f}')
    print(f'    个股总盈利:   {(last_px/entry_px-1)*100:+.1f}% ({(last_px-entry_px)*shares:+,.0f})')

cash = today_val - total_pos_value
print(f'\n【资金分配】')
print(f'  持仓市值: {total_pos_value:,.0f} ({total_pos_value/today_val*100:.1f}%)')
print(f'  现金余额: {cash:,.0f} ({cash/today_val*100:.1f}%)')

tl = e.export_trade_log()
today_str = str(today.date())
today_trades = tl[tl['date'].astype(str).str[:10] == today_str]
if len(today_trades[today_trades['shares'] > 0]) > 0:
    print(f'\n【今日交易】')
    for _, t in today_trades[today_trades['shares'] > 0].iterrows():
        print(f'  {t["action"]} {t["symbol"]} {int(t["shares"])}股 @{t["price"]:.2f}  {t["reason"]}')

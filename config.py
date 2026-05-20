#!/usr/bin/env python3
"""B2 Spring策略 — Step1+Step4 TPE优化参数 (2026-05-20 精简版)"""

# ============================================================
# B2 入场 + 权重 (Step1 TPE优化 + Step4 动态信号有效期)
# ============================================================
BEST_B2 = {
    # 入场阈值 (固定, 不进搜索)
    'j_prev_max': 20, 'gain_min': 0.04, 'j_today_max': 65,
    'shadow_max': 0.035, 'j_today_loose': 70, 'shadow_loose': 0.045,
    'prior_strong_ret': 0.20,
    # 权重倍数 (Step1 TPE搜索最优)
    'weight_gain_2x_thresh': 0.0842, 'weight_gain_2x': 1.479,
    'weight_gap_thresh': 0.02, 'weight_gap_up': 1.5,
    'weight_shrink': 1.171, 'weight_deep_shrink_ratio': 0.80,
    'weight_deep_shrink': 1.3, 'weight_pullback_thresh': -0.05,
    'weight_pullback': 1.2, 'weight_shadow_discount_thresh': 0.015,
    'weight_shadow_discount': 0.7, 'weight_strong_discount': 0.8,
    'weight_brick_resonance': 1.5,
    # 信号排序
    't1_threshold': 3.5, 't1_multiplier': 1.5,
    # 动态信号有效期 (Step4 TPE搜索最优)
    'sig_expire_thresh': 0.973, 'max_signal_days': 14,
    'vol_check_days': 5, 'vol_spike_thresh': 1.954,
    # 趋势过滤
    'trend_filter': True,
}

# ============================================================
# 离场/止损 (Step1 TPE搜索最优, 无breakeven)
# ============================================================
FROZEN_STOP_DEFAULTS = {
    'trailing_activate': 0.1457,   # 移动止盈激活
    'trailing_pct': 0.0410,        # 移动止盈回撤
    'peak_dd_stop': -0.1904,       # 峰值回撤硬止损
    'hard_stop': {'aggressive': -0.18, 'normal': -0.10, 'defensive': -0.08},
    'hard_stop_min_days': 5,
    'take_profit_full': 0.60, 'take_profit_partial_frac': 0.50,
    'time_stop_loser_days': 20, 'time_stop_loser_pct': -0.05,
    'checkpoint_60d_min_ret': 0.05, 'max_hold_days': 180,
}

# ============================================================
# 引擎配置
# ============================================================
ENGINE_CONFIG = {
    'initial_capital': 100_000, 'max_positions': 4, 'min_positions': 0,
    'max_entries_per_day': 3, 'single_position_cap': 0.50,
    'min_single_amount': 8000, 'commission_buy': 0.0003,
    'commission_sell': 0.0003, 'lot_size': 100,
}

# ============================================================
# 训练窗口
# ============================================================
TRAIN_WINDOWS = [('N_Bull', '2024-01-01', '2025-09-30')]
VAL_WINDOW = ('Val_2025Q4', '2025-10-01', '2025-12-31')
TEST_WINDOW = ('Test_2026', '2026-01-01', 'today')
BEAR_STRESS = ('Bear_2018', '2018-01-01', '2018-12-31')

# ============================================================
# TPE搜索空间 (Step1+Step4已搜索完成, 保留供后续使用)
# ============================================================
TPE_PARAM_SPACE = {
    'trailing_activate':       (0.08, 0.25),
    'trailing_pct':            (0.02, 0.12),
    'peak_dd_stop':            (-0.30, -0.08),
    'weight_gain_2x_thresh':   (0.05, 0.15),
    'weight_gain_2x':          (1.1, 3.5),
    'weight_shrink':           (0.8, 1.8),
    'sig_expire_thresh':       (0.90, 0.99),
    'max_signal_days':         (5, 21),
    'vol_check_days':          (3, 10),
    'vol_spike_thresh':        (1.2, 3.0),
}

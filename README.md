# B2 Spring -- A股短线反转策略

KDJ + MACD + 砖型图 + 白黄线多指标共振，专注 Spring 超跌反弹信号。

## 回测结果

| 窗口 | 收益 | Sharpe | 胜率 | 最大回撤 |
|------|------|--------|------|----------|
| 2026 YTD (01-01 ~ 05-21) | +182.4% | 4.427 | 65.2% | -7.1% |

## 策略架构

```
8条件AND入场 -> 11层权重叠加 -> P50-P96市值过滤 -> T+1开盘买入
                                            |
        5条离场规则 <- 3持仓(rotate_base=0.5) <- 信号强度排序
```

### 入场 (8条件AND)
① J<20 ② 涨幅>4% ③ J<65 ④ 放量 ⑤ 上影<3.5% ⑥ 白>黄 ⑦ 收>黄 ⑧ 上影<振幅1/4

### 离场 (5条优先级)
1. **take_half**: +10% 卖一半锁利润
2. **white_line**: 跌破白线 (双作用: 截损+牵牛绳)
3. **yellow_line**: 跌破黄线
4. **candle**: 跌破买入日最低价
5. **time_stop**: 持仓4天不涨

### 权重叠加 (11层, 上限5x)
涨幅>9% ×2.5 | 跳空 ×1.5 | 缩量 ×1.2 | 深度缩量 ×1.3 | 回调 ×1.2 | 砖型共振 ×1.5 | 上影折价 ×0.7

## 项目结构

```
├── run_backtest_2026.py         # 回测入口
├── strategy_spring.py           # Spring信号引擎
├── close_sell_check.py          # 收盘卖出检测
├── config.py                    # 参数配置
├── indicators.py                # 技术指标 (KDJ/MACD/砖型图/白黄线)
├── oamv.py                      # 活筹指数
├── grid_search.py               # Optuna TPE参数优化
├── generate_report_xlsx.py      # Excel报告
└── ML_optimization/
    ├── phase2c_oamv_grid_search.py  # 仿真引擎
    ├── phase2c_bull_grid_search.py  # 数据加载
    └── mktcap_utils.py              # 市值过滤
```

## 使用方法

```bash
python run_backtest_2026.py       # 回测年初至今

# 信号输出保存到 output/signals/, T+1开盘执行买入
# 收盘卖出检测: python close_sell_check.py (5条离场与引擎一致)
```

## 交易规则
- T日收盘信号 -> T+1开盘买入
- 单票上限50%, 最多4只, 单日最多3笔, 单笔>=8000
- 买入佣金0.03%, 卖出0.13%

## 免责声明
仅供研究学习，不构成投资建议。历史回测不代表未来收益。

"""
动量轮动回测（新版，适用于较新 Python 环境）

说明：
- 使用 N 日变动率（ROC）作为动量指标，按固定周期轮动持仓，交易在次日开盘价执行；
- 支持在线数据（`yfinance`）与本地 CSV 收盘价数据；生成绩效指标与可视化仪表板。
"""
import os
import sys
import math
import json
from typing import List, Optional, Tuple, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib as mpl
import seaborn as sns


PROJECT_NAME = "MomentumRotationBacktest"

mpl.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS', 'Noto Sans CJK SC']
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['axes.unicode_minus'] = False

CN_NAMES = {
    "AAPL": "苹果",
    "GOOGL": "谷歌",
    "MSFT": "微软",
    "TSLA": "特斯拉",
    "NVDA": "英伟达",
    "META": "Meta",
    "AMZN": "亚马逊",
    "NFLX": "奈飞",
}

def _to_cn_name(sym):
    s = str(sym)
    if s in CN_NAMES:
        return CN_NAMES[s]
    return (u"代码" + s) if s.isdigit() else s


def get_data(symbol_list: List[str], start: str, end: str, use_local_csv: Optional[str] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    加载开盘/收盘价数据；优先本地 CSV，否则使用 `yfinance` 下载。

    返回 `(price_open_df, price_close_df)`，按日期索引对齐。
    """
    price_open_df = None
    price_close_df = None
    if use_local_csv and os.path.isfile(use_local_csv):
        df = pd.read_csv(use_local_csv, index_col=0, parse_dates=True)
        df = df.sort_index()
        price_close_df = df
        price_open_df = None
    else:
        try:
            import yfinance as yf
        except Exception as e:
            raise RuntimeError("yfinance is required when no local CSV is provided") from e
        data = yf.download(symbol_list, start=start, end=end, auto_adjust=False, progress=False)
        if isinstance(data.columns, pd.MultiIndex):
            price_open_df = data["Open"].copy()
            price_close_df = data["Close"].copy()
        else:
            price_open_df = data[["Open"]].rename(columns={"Open": symbol_list[0]})
            price_close_df = data[["Close"]].rename(columns={"Close": symbol_list[0]})
        price_open_df = price_open_df.dropna(how="all").sort_index()
        price_close_df = price_close_df.dropna(how="all").sort_index()
    if price_open_df is not None:
        idx = price_open_df.index
        price_close_df = price_close_df.reindex(idx).ffill()
    else:
        idx = price_close_df.index
    return price_open_df, price_close_df


def compute_momentum(price_close_df: pd.DataFrame, lookback: int) -> pd.DataFrame:
    """计算 N 日 ROC：`Close(t)/Close(t-N)-1`"""
    return price_close_df / price_close_df.shift(lookback) - 1.0


def make_rebalance_schedule(index: pd.DatetimeIndex, lookback: int, holding: int) -> List[pd.Timestamp]:
    """从满足观察期后开始，每隔 `holding` 天生成一个调仓日"""
    start_pos = lookback
    dates = list(index[start_pos:])
    if not dates:
        return []
    schedule = [dates[0]]
    pos = start_pos
    while pos < len(index):
        pos += holding
        if pos < len(index):
            schedule.append(index[pos])
    return schedule


def select_weights(momentum_df: pd.DataFrame, schedule: List[pd.Timestamp], top_k: int) -> pd.DataFrame:
    """每个调仓日选择动量前 `top_k` 等权持仓；权重自次日生效，至下次调仓"""
    assets = list(momentum_df.columns)
    weights = pd.DataFrame(0.0, index=momentum_df.index, columns=assets)
    for i, d in enumerate(schedule):
        row = momentum_df.loc[d]
        row = row.dropna()
        if len(row) == 0:
            continue
        k = min(top_k, len(row))
        selected = list(row.sort_values(ascending=False).iloc[:k].index)
        w = 1.0 / float(k)
        start_idx = momentum_df.index.get_loc(d) + 1
        if start_idx >= len(momentum_df.index):
            continue
        if i + 1 < len(schedule):
            end_date = schedule[i + 1]
            end_idx = momentum_df.index.get_loc(end_date)
        else:
            end_idx = len(momentum_df.index)
        period_index = momentum_df.index[start_idx:end_idx]
        for s in selected:
            weights.loc[period_index, s] = w
    return weights


def compute_returns(price_df: pd.DataFrame) -> pd.DataFrame:
    """计算 `pct_change()` 并右移一日，使权重与次日收益匹配"""
    return price_df.pct_change().shift(-1)


def backtest(weights_df: pd.DataFrame, returns_df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
    """合成组合日收益与净值曲线"""
    aligned_returns = returns_df.reindex(weights_df.index)
    portfolio_ret = (weights_df * aligned_returns).sum(axis=1)
    portfolio_ret = portfolio_ret.dropna()
    nav = (1.0 + portfolio_ret).cumprod()
    return portfolio_ret, nav


def benchmark_equal_weight(returns_df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
    """构建等权持有所有资产的基准组合"""
    n = returns_df.shape[1]
    if n == 0:
        return pd.Series(dtype=float), pd.Series(dtype=float)
    bench_weights = pd.DataFrame(1.0 / float(n), index=returns_df.index, columns=returns_df.columns)
    bench_ret = (bench_weights * returns_df).sum(axis=1).dropna()
    bench_nav = (1.0 + bench_ret).cumprod()
    return bench_ret, bench_nav


def max_drawdown(nav: pd.Series) -> Tuple[float, Optional[pd.Timestamp], Optional[pd.Timestamp]]:
    """计算最大回撤及区间"""
    cum_max = nav.cummax()
    drawdown = nav / cum_max - 1.0
    mdd = drawdown.min() if len(drawdown) else np.nan
    if len(drawdown) == 0 or np.isnan(mdd):
        return np.nan, None, None
    end_date = drawdown.idxmin()
    start_mask = nav.loc[:end_date]
    start_date = start_mask.idxmax()
    return float(mdd), start_date, end_date


def regression_alpha_beta(strategy_ret: pd.Series, bench_ret: pd.Series) -> Tuple[float, float]:
    """通过矩估计计算 Alpha（年化）与 Beta"""
    df = pd.DataFrame({"y": strategy_ret, "x": bench_ret}).dropna()
    if len(df) < 2:
        return np.nan, np.nan
    x = df["x"].values
    y = df["y"].values
    beta = np.cov(x, y)[0, 1] / np.var(x)
    alpha_daily = y.mean() - beta * x.mean()
    alpha_annual = alpha_daily * 252.0
    return float(alpha_annual), float(beta)


def metrics(strategy_ret: pd.Series, bench_ret: pd.Series, nav: pd.Series) -> Dict[str, float]:
    """汇总总/年化收益、年化波动、夏普、最大回撤、Alpha/Beta"""
    ret = strategy_ret.dropna()
    bench = bench_ret.dropna()
    ann_ret = float(np.prod(1.0 + ret) ** (252.0 / len(ret)) - 1.0) if len(ret) else np.nan
    ann_vol = float(ret.std() * math.sqrt(252.0)) if len(ret) else np.nan
    sharpe = float(ann_ret / ann_vol) if ann_vol and not np.isnan(ann_vol) and ann_vol != 0.0 else np.nan
    mdd, mdd_start, mdd_end = max_drawdown(nav)
    alpha, beta = regression_alpha_beta(ret, bench)
    return {
        "total_return": float(nav.iloc[-1] - 1.0) if len(nav) else np.nan,
        "annual_return": ann_ret,
        "annual_volatility": ann_vol,
        "sharpe_ratio": sharpe,
        "max_drawdown": mdd,
        "max_drawdown_start": str(mdd_start) if mdd_start is not None else "",
        "max_drawdown_end": str(mdd_end) if mdd_end is not None else "",
        "alpha": alpha,
        "beta": beta,
    }


def monthly_returns(ret: pd.Series) -> pd.DataFrame:
    """计算月度复合收益并返回（年×月）透视表"""
    df = ret.to_frame("ret").dropna()
    df["month"] = df.index.to_period("M")
    grouped = df.groupby("month").apply(lambda x: float(np.prod(1.0 + x["ret"]) - 1.0))
    out = grouped.to_timestamp()
    out_df = out.to_frame("ret")
    out_df["year"] = out_df.index.year
    out_df["mon"] = out_df.index.month
    pivot = out_df.pivot(index="year", columns="mon", values="ret").sort_index()
    return pivot


def monthly_win_rate_vs_bench(strategy_ret: pd.Series, bench_ret: pd.Series) -> float:
    """相对基准的月度胜率（策略月收益 > 基准月收益 的比例）"""
    s = monthly_returns(strategy_ret)
    b = monthly_returns(bench_ret)
    df = pd.DataFrame({"s": s.stack(), "b": b.stack()}).dropna()
    if len(df) == 0:
        return np.nan
    wins = (df["s"] > df["b"]).sum()
    return float(wins) / float(len(df))


def plot_dashboard(nav: pd.Series, bench_nav: pd.Series, strategy_ret: pd.Series, bench_ret: pd.Series, metrics_dict: Dict[str, float], weights_df: pd.DataFrame, out_path: str) -> None:
    """绘制 3×2 仪表板：净值/超额、回撤、月度热力图、持仓、年度收益、滚动指标"""
    fig, axes = plt.subplots(3, 2, figsize=(16, 12))
    ax1 = axes[0, 0]
    ax2 = axes[0, 1]
    ax3 = axes[1, 0]
    ax4 = axes[1, 1]
    ax5 = axes[2, 0]
    ax6 = axes[2, 1]

    ax1.plot(nav.index, nav.values, label="策略净值")
    ax1.plot(bench_nav.index, bench_nav.values, label="基准净值")
    ann = metrics_dict['annual_return']
    sh = metrics_dict['sharpe_ratio']
    sh_text = ("{:.2f}".format(sh)) if not np.isnan(sh) else "nan"
    ax1.set_title("净值（年化收益={:.2%}, 夏普={}）".format(ann, sh_text))
    ax1.set_ylabel("净值")
    ax1.legend()
    excess = nav.reindex(bench_nav.index) / bench_nav - 1.0
    ax1b = ax1.twinx()
    ax1b.plot(excess.index, excess.values, color="gray", alpha=0.5, label="超额收益")
    ax1b.set_ylabel("超额收益")

    cum_max = nav.cummax()
    drawdown = nav / cum_max - 1.0
    valid_idx = pd.to_datetime(drawdown.index, errors='coerce')
    mask = (~pd.isnull(valid_idx)) & (valid_idx >= pd.Timestamp('1990-01-01')) & (valid_idx <= pd.Timestamp('2100-12-31'))
    x_dates = valid_idx[mask]
    y_vals = drawdown.values[mask]
    ax2.plot(x_dates, y_vals, color="crimson")
    ax2.set_title("回撤（最小={:.2%}）".format(drawdown.min()))
    ax2.set_ylabel("回撤")
    ax2.set_xlabel("时间（日期）")
    n_ticks = min(8, len(x_dates)) if len(x_dates) > 0 else 0
    idxs = np.linspace(0, max(len(x_dates) - 1, 0), n_ticks, dtype=int) if n_ticks > 0 else np.array([])
    tick_dates = x_dates.values[idxs] if n_ticks > 0 else []
    ax2.set_xticks(tick_dates)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax2.minorticks_off()
    ax2.tick_params(axis='x', labelsize=8)

    mr = monthly_returns(strategy_ret)
    sns.heatmap(mr, ax=ax3, cmap="RdYlGn", center=0.0, annot=True, fmt=".1%")
    ax3.set_title("月度收益")

    weights_cn = weights_df.copy()
    weights_cn.columns = [ _to_cn_name(c) for c in weights_cn.columns ]
    weights_cn.plot.area(ax=ax4, stacked=True)
    ax4.set_title("持仓权重")
    ax4.set_ylabel("权重")

    s_year = monthly_returns(strategy_ret).mean(axis=1)
    b_year = monthly_returns(bench_ret).mean(axis=1)
    common_years = sorted(set(s_year.index).intersection(set(b_year.index)))
    s_vals = [s_year.loc[y] for y in common_years]
    b_vals = [b_year.loc[y] for y in common_years]
    x = np.arange(len(common_years))
    width = 0.35
    ax5.bar(x - width / 2, s_vals, width, label="策略")
    ax5.bar(x + width / 2, b_vals, width, label="基准")
    ax5.set_xticks(x)
    ax5.set_xticklabels([str(y) for y in common_years])
    ax5.set_title("年度收益")
    ax5.legend()

    roll_mean = strategy_ret.rolling(252).mean()
    roll_std = strategy_ret.rolling(252).std()
    roll_sharpe = np.sqrt(252.0) * roll_mean / roll_std
    roll_vol = roll_std * np.sqrt(252.0)
    ax6.plot(roll_sharpe.index, roll_sharpe.values, label="滚动夏普")
    ax6.plot(roll_vol.index, roll_vol.values, label="滚动波动率")
    ax6.set_title("滚动指标")
    ax6.legend()

    plt.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def build_rebalance_log(schedule: List[pd.Timestamp], momentum_df: pd.DataFrame, weights_df: pd.DataFrame, top_k: int) -> pd.DataFrame:
    """生成调仓日志：记录动量、入选资产与当期权重（权重次日生效）"""
    records = []
    for i, d in enumerate(schedule):
        if d not in momentum_df.index:
            continue
        row = momentum_df.loc[d]
        row_sorted = row.sort_values(ascending=False)
        selected = list(row_sorted.dropna().index[:min(top_k, len(row_sorted.dropna()))])
        w_row = weights_df.loc[weights_df.index[weights_df.index.get_loc(d) + 1]] if (weights_df.index.get_loc(d) + 1) < len(weights_df.index) else weights_df.loc[d]
        record = {
            "日期": str(d),
            "入选资产": json.dumps([ _to_cn_name(x) for x in selected ]),
        }
        for a in momentum_df.columns:
            cn = _to_cn_name(a)
            record[u"动量_{}".format(cn)] = float(row[a]) if not pd.isna(row[a]) else np.nan
            record[u"权重_{}".format(cn)] = float(w_row[a]) if not pd.isna(w_row[a]) else 0.0
        records.append(record)
    return pd.DataFrame(records)


def main():
    """主流程：加载数据 → 计算动量/权重 → 回测 → 统计指标 → 输出"""
    symbol_list = ["AAPL", "GOOGL", "MSFT", "TSLA", "NVDA", "META", "AMZN", "NFLX"]
    start_date = "2020-01-01"
    end_date = "2023-12-31"
    lookback_period = 20
    holding_period = 20
    top_k = 3
    initial_capital = 10000.0
    use_local_csv = None

    price_open_df, price_close_df = get_data(symbol_list, start_date, end_date, use_local_csv)
    base_index = price_close_df.index if price_open_df is None else price_open_df.index
    momentum_df = compute_momentum(price_close_df.reindex(base_index), lookback_period)
    schedule = make_rebalance_schedule(base_index, lookback_period, holding_period)
    weights_df = select_weights(momentum_df, schedule, top_k)
    if price_open_df is not None:
        returns_df = compute_returns(price_open_df.reindex(base_index))
    else:
        returns_df = compute_returns(price_close_df.reindex(base_index))
    strategy_ret, nav = backtest(weights_df, returns_df)
    bench_ret, bench_nav = benchmark_equal_weight(returns_df)
    m = metrics(strategy_ret, bench_ret, nav)
    monthly_win = monthly_win_rate_vs_bench(strategy_ret, bench_ret)
    m["monthly_win_rate_vs_bench"] = monthly_win

    out_dir = os.path.dirname(os.path.abspath(__file__))
    log_df = build_rebalance_log(schedule, momentum_df, weights_df, top_k)
    log_path = os.path.join(out_dir, "rebalance_log.csv")
    log_df.to_csv(log_path, index=False)
    dashboard_path = os.path.join(out_dir, "dashboard.png")
    plot_dashboard(nav, bench_nav, strategy_ret, bench_ret, m, weights_df, dashboard_path)

    print(json.dumps(m, ensure_ascii=False, indent=2))
    print("Saved: {}".format(log_path))
    print("Saved: {}".format(dashboard_path))

    # 可选：本地示例 CSV 列名汉化（若存在）
    price_csv = os.path.join(out_dir, "price_data.csv")
    try:
        if os.path.isfile(price_csv):
            df = pd.read_csv(price_csv, index_col=0, parse_dates=True)
            df.columns = [ _to_cn_name(c) for c in df.columns ]
            df.index.name = u"日期"
            df.to_csv(price_csv)
    except Exception:
        pass


if __name__ == "__main__":
    main()

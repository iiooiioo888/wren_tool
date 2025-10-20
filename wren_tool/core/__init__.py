"""
Wren Tool 核心模組
提供增強版交易策略回測功能
"""

from .backtester import (
    load_data,
    generate_signals,
    generate_sma_signals,
    generate_rsi_signals,
    simulate_trades_enhanced,
    run_poc,
    run_benchmark
)

from .risk_metrics import (
    calculate_max_drawdown,
    calculate_max_drawdown_from_equity_curve,
    calculate_equity_curve,
    calculate_sharpe_ratio,
    calculate_sortino_ratio,
    calculate_calmar_ratio,
    calculate_volatility,
    calculate_downside_volatility,
    get_risk_report
)

__all__ = [
    # 數據載入和處理
    'load_data',
    'generate_signals',
    'generate_sma_signals',
    'generate_rsi_signals',

    # 回測引擎
    'simulate_trades_enhanced',
    'run_poc',
    'run_benchmark',

    # 風險指標
    'calculate_max_drawdown',
    'calculate_max_drawdown_from_equity_curve',
    'calculate_equity_curve',
    'calculate_sharpe_ratio',
    'calculate_sortino_ratio',
    'calculate_calmar_ratio',
    'calculate_volatility',
    'calculate_downside_volatility',
    'get_risk_report'
]

__version__ = "2.0.0"

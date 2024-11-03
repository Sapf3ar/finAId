from .plots import (
    create_bar_plot,
    create_line_plot,
    create_pie_chart,
    create_scatter_plot,
)
from typing import List
from loguru import logger


class Profile:
    metrics: List[str] = []

    def _code2metric(self) -> List[str]:
        calc_metrics = []
        for metric in self.metrics:
            long_code = name2code(metric, reverse=True)
            calc_metrics.append(long_code)
        return calc_metrics


def profile_name2type(x: str) -> type[Profile]:
    ind = [
        "Инвестор в Фонде",
        "Индивидуальный инвестор",
        "Кредитный аналитик",
        "Финансист в корпорации",
    ].index(x)
    logger.warning(ind)
    match ind:
        case 0:
            return VCInvest
        case 1:
            return PerInvest
        case 2:
            return CreditAnalyst
        case 3:
            return CorpAnalyst
        case _:
            return Profile


def parse_metrics(query) -> List[str]:
    metrics = []
    for profile in query.profiles:
        metrics.extend(profile_name2type(profile)()._code2metric())
    for i in range(len(metrics)):
        metrics[i] = name2code(metrics[i])
    return metrics


class PerInvest(Profile):
    metrics: List[str] = ["RoE", "RoA", "DY", "FCF"]


class VCInvest(Profile):
    metrics: List[str] = ["EV/EBITDA", "RGR", "GM", "ROIC", "RevMult"]


class CreditAnalyst(Profile):
    metrics: List[str] = ["D/E", "ICR", "FCF", "CR", "QR", "ROIC"]


class CorpAnalyst(Profile):
    metrics: List[str] = ["NPM", "EM", "CFO", "DSCR", "CR", "RoA"]


def name2code(key: str, reverse: bool = False) -> str:
    short_metrics = {
        "Debt-to-Equity Ratio": "D/E",
        "Current Ratio": "CR",
        "Quick Ratio": "QR",
        "Interest Coverage Ratio": "ICR",
        "Free Cash Flow": "FCF",
        "Net Profit Margin": "NPM",
        "EBITDA Margin": "EM",
        "Cash Flow from Operations (CFO)": "CFO",
        "Debt Service Coverage Ratio": "DSCR",
        "Sharpe Ratio": "SR",
        "Return on Equity (ROE)": "RoE",
        "Return on Assets (ROA)": "RoA",
        "Dividend Yield": "DY",
        "Revenue Multiple": "RevMult",
        "EV/EBITDA": "EV/EBITDA",
        "Revenue Growth Rate": "RGR",
        "Gross Margin": "GM",
        "Return on Invested Capital": "ROIC",
    }
    if reverse:
        long_metrics = {v: k for k, v in short_metrics.items()}
        return long_metrics[key]
    return short_metrics[key]


def metric2chart(base=False):
    if base:
        charts = ["bar", "line", "pie", "scatter"]
    else:
        charts = [
            create_bar_plot,
            create_line_plot,
            create_pie_chart,
            create_scatter_plot,
        ]
    chart_recommendations = {
        "D/E": {"single_year": 0, "multiple_years": 1},  # Debt-to-Equity Ratio
        "CR": {"single_year": 0, "multiple_years": 1},  # Current Ratio
        "QR": {"single_year": 0, "multiple_years": 1},  # Quick Ratio
        "ICR": {"single_year": 0, "multiple_years": 1},  # Interest Coverage Ratio
        "FCF": {"single_year": 0, "multiple_years": 1},  # Free Cash Flow
        "NPM": {"single_year": 0, "multiple_years": 1},  # Net Profit Margin
        "EM": {"single_year": 0, "multiple_years": 1},  # EBITDA Margin
        "CFO": {"single_year": 0, "multiple_years": 1},  # Cash Flow from Operations
        "DSCR": {"single_year": 0, "multiple_years": 1},  # Debt Service Coverage Ratio
        "SR": {
            "single_year": 0,
            "multiple_years": 3,
        },  # Sharpe Ratio (scatter for correlation)
        "RoE": {"single_year": 0, "multiple_years": 1},  # Return on Equity
        "RoA": {"single_year": 0, "multiple_years": 1},  # Return on Assets
        "DY": {"single_year": 0, "multiple_years": 1},  # Dividend Yield
        "RevMult": {"single_year": 0, "multiple_years": 1},  # Revenue Multiple
        "EV/EBITDA": {"single_year": 0, "multiple_years": 1},  # EV/EBITDA
        "RGR": {
            "single_year": 1,
            "multiple_years": 1,
        },  # Revenue Growth Rate (line for trends)
        "GM": {"single_year": 0, "multiple_years": 1},  # Gross Margin
        "ROIC": {"single_year": 0, "multiple_years": 1},  # Return on Invested Capital
    }
    for key, value in chart_recommendations.items():
        for k, v in value.items():
            chart_recommendations[key][k] = charts[v]
    return chart_recommendations


# 1. Debt-to-Equity Ratio
def debt_to_equity(financials):
    return financials["общие_обязательства"] / financials["общий_капитал"]


# 2. Current Ratio
def current_ratio(financials):
    return financials["текущие_активы"] / financials["текущие_обязательства"]


# 3. Quick Ratio
def quick_ratio(financials):
    return (financials["текущие_активы"] - financials["запасы"]) / financials[
        "текущие_обязательства"
    ]


# 4. Interest Coverage Ratio
def interest_coverage(financials):
    return (
        financials["прибыль_до_налогов_и_процентов (EBIT)"]
        / financials["процентные_расходы"]
    )


# 5. Free Cash Flow
def free_cash_flow(financials):
    return (
        financials["денежный_поток_от_операций"] - financials["выплаченные_дивиденды"]
    )


# 6. Net Profit Margin
def net_profit_margin(financials):
    return financials["чистая_прибыль"] / financials["выручка"]


# 7. EBITDA Margin
def ebitda_margin(financials):
    return (
        financials["прибыль_до_налогов_процентов_и_амортизации (EBITDA)"]
        / financials["выручка"]
    )


# 8. Cash Flow from Operations (CFO) - This is given directly
def cash_flow_operations(financials):
    return financials["денежный_поток_от_операций"]


# 9. Debt Service Coverage Ratio
def debt_service_coverage_ratio(financials):
    return financials["денежный_поток_от_операций"] / financials["общий_долг"]


# 10. Sharpe Ratio
def sharpe_ratio(financials):
    return (
        financials["ожидаемая_доходность"] - financials["безрисковая_ставка"]
    ) / financials["стандартное_отклонение_доходности"]


# 11. Return on Equity (ROE)
def return_on_equity(financials):
    return financials["чистая_прибыль"] / financials["средний_капитал"]


# 12. Return on Assets (ROA)
def return_on_assets(financials):
    return financials["чистая_прибыль"] / financials["средние_активы"]


# 13. Dividend Yield
def dividend_yield(financials):
    return financials["дивиденды_на_акцию"] / financials["цена_акции"]


# 14. Revenue Multiple
def revenue_multiple(financials):
    return financials["стоимость_предприятия (EV)"] / financials["выручка"]


# 15. EV/EBITDA
def ev_ebitda(financials):
    return (
        financials["стоимость_предприятия (EV)"]
        / financials["прибыль_до_налогов_процентов_и_амортизации (EBITDA)"]
    )


# 16. Revenue Growth Rate
def revenue_growth_rate(financials):
    return (financials["выручка"] - financials["выручка_за_прошлый_год"]) / financials[
        "выручка_за_прошлый_год"
    ]


# 17. Gross Margin
def gross_margin(financials):
    return financials["валовая_прибыль"] / financials["выручка"]


# 18. Return on Invested Capital (ROIC)
def return_on_invested_capital(financials):
    return (
        financials["прибыль_до_налогов_и_процентов (EBIT)"]
        / financials["вложенный_капитал"]
    )


def calculate_metric_value(financials):
    import numpy as np

    metric = {
        "Debt-to-Equity Ratio": debt_to_equity(financials),
        "Current Ratio": current_ratio(financials),
        "Quick Ratio": quick_ratio(financials),
        "Interest Coverage Ratio": interest_coverage(financials),
        "Free Cash Flow": free_cash_flow(financials),
        "Net Profit Margin": net_profit_margin(financials),
        "EBITDA Margin": ebitda_margin(financials),
        "Cash Flow from Operations (CFO)": cash_flow_operations(financials),
        "Debt Service Coverage Ratio": debt_service_coverage_ratio(financials),
        "Sharpe Ratio": sharpe_ratio(financials),
        "Return on Equity (ROE)": return_on_equity(financials),
        "Return on Assets (ROA)": return_on_assets(financials),
        "Dividend Yield": dividend_yield(financials),
        "Revenue Multiple": revenue_multiple(financials),
        "EV/EBITDA": ev_ebitda(financials),
        "Revenue Growth Rate": revenue_growth_rate(financials),
        "Gross Margin": gross_margin(financials),
        "Return on Invested Capital": return_on_invested_capital(financials),
    }

    return {name2code(k): v + np.random.randn() for k, v in metric.items()}


# Define groups with metric keys


# Function to map short metric names to group names
def group_metrics(short_metrics):
    # Reverse the groups dictionary for lookup
    groups = {
        "Liquidity": ["CR", "QR", "CFO"],
        "Leverage": ["D/E", "ICR", "DSCR"],
        "Profitability": ["NPM", "EM", "GM", "RoE", "RoA", "ROIC"],
        "Valuation": ["DY", "RevMult", "EV/EBITDA"],
        "Risk-Adjusted Return": ["SR"],
        "Growth": [
            "RGR",
            "EPSG",
            "DGR",
            "CapExG",
            "SRevenueG",
            "FCFG",
        ],  # Adding EPSG, DGR as example growth metrics
    }
    groups = {
        "Liquidity": ["CR", "QR"],
        "Leverage": ["D/E", "ICR", "DSCR"],
        "Profitability": ["NPM", "EM", "GM", "RoE", "RoA", "ROIC"],
        "Valuation": ["DY", "RevMult", "EV/EBITDA"],
        "Growth": ["RGR"],
        "Cash Flow": ["FCF", "CFO"],
        "Risk-Adjusted Return": ["SR"],
        "Efficiency": [
            "ROIC",
            "RoE",
            "RoA",
        ],  # ROIC added to efficiency since it measures capital utilization
    }
    grouped_metrics = {group: [] for group in groups}

    # Create a reverse lookup dictionary for metrics
    metric_to_group = {}
    for group_name, metric_keys in groups.items():
        for metric_key in metric_keys:
            metric_to_group[metric_key] = group_name

    # Map each short metric to its corresponding group
    for short_name in short_metrics:
        group_name = metric_to_group.get(short_name, "Uncategorized")
        if group_name in grouped_metrics:
            grouped_metrics[group_name].append(short_name)
        else:
            grouped_metrics["Uncategorized"] = grouped_metrics.get(
                "Uncategorized", []
            ) + [short_name]

    return grouped_metrics

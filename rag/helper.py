from loguru import logger
import pandas as pd
from .plots import plot_multiple_metrics, single_bar
from .metrics import metric2chart, parse_metrics, calculate_metric_value, group_metrics
from typing import Generator, List, Optional, Callable
import time
from dataclasses import dataclass


@dataclass
class RagQuery:
    prompt: str
    profiles: List[str]
    metrics: Optional[List[Callable[[List[float]], List[float]]]] = None
    plots: Optional[List[Callable[List[float], float]]] = None
    base_plots: Optional[List[Callable[List[float], float]]] = None
    base_metrics: Optional[List[Callable[List[float], float]]] = None
    data: Optional[pd.DataFrame] = None
    text_output: Optional[str] = None
    years: Optional[List[int]] = None
    companies: Optional[List[str]] = None


def llm_response(
    prompt: str,
    agent_manager,
) -> dict[str, (List[int] | List[str] | pd.DataFrame)]:
    """ 
    Response type:
    [
    {
        "Company": {
                    "Year": {
                        "Metric_name":metric_value
                        }
                    }
    },
    {
        "Tags": {"metric_names":["metric_1", "metric_2"]},
        "Text":["Text output from llm"]
    }
    ]
    """

    financials_per_comp: dict[str, int | float] = agent_manager.financials
    
    # metrics_year = financials_year["yearly"]
    frames = []

    for comp, metrics_year in financials_comp.items():
        base_dict = {}
        base_dict["year"] = []
        for year, metrics in metrics_year.items():
            base_dict["year"].append(year)
            
            for k, v in metrics.items():
                cur_metric = base_dict.get(k, [])
                cur_metric.append(v)
                base_dict[k] = cur_metric
        base_dict["comp"] = [comp] * len(base_dict["year"])

        frame = pd.DataFrame(base_dict)
        frames.append(frame)

    output = {"data": pd.concat(frames)}
    # relevant_metrics: List[str] = ["цена_акции", "выручка_за_прошлый_год"]
    relevant_metrics: List[str] = ["COVID-19"]
    plots: List[int] = [0, 2]
    
    out = eval(agent_manager.response(prompt))
    text: List[str] = [out['rag_response']]
    output["base_metrics"] = relevant_metrics
    output["base_plots"] = out['for_plot']['relevant_plots']
    output["text"] = text
    return output


def compound_metrics(df: pd.DataFrame) -> pd.DataFrame:
    tot_metrics_comp = []
    for name, data in df.groupby("comp"):
        tot_metrics = []
        for ind, row in data.iterrows():
            metrics = calculate_metric_value(row)
            metrics["year"] = row.year
            tot_metrics.append(metrics)

        tot_metrics = pd.DataFrame(tot_metrics)
        # tot_metrics = tot_metrics.merge(df)
        tot_metrics["comp"] = [name] * len(tot_metrics)
        tot_metrics_comp.append(tot_metrics)
    tot_metrics = pd.concat(tot_metrics_comp)
    tot_metrics = df.merge(tot_metrics)

    tot_metrics_comp[0].to_csv("test.csv", index=False)

    return tot_metrics


def parse_plots(query: RagQuery) -> RagQuery:
    plots = [metric2chart(base=True)[metric] for metric in query.metrics]

    base_plots = [metric2chart(base=True)["RoE"] for _ in query.base_metrics]
    query.base_plots = base_plots
    query.plots = plots
    return query


def get_rag(query: RagQuery, agent_manager) -> RagQuery:
    query.metrics = parse_metrics(query)
    output = llm_response(query.prompt, agent_manager=agent_manager)
    query.data = output["data"]
    query.companies = query.data.comp.unique().tolist()
    query.text_output = output["text"]
    query.base_metrics = output["base_metrics"]
    logger.debug(query.data.year)
    query.data = compound_metrics(query.data)
    logger.debug(query.data.year)
    query = parse_plots(query)
    query.years = query.data.year.values
    query = draw_plots(query)
    return query


def draw_plots(query: RagQuery) -> RagQuery:
    base_plots = []
    logger.debug(query.data.year)
    plot_name = query.base_plots[0]["multiple_years"]
    for metric in query.base_metrics:
        fig = single_bar(
            data=query.data,
            x="year",
            y=metric,
            title="Suggested plots",
            company_column="comp",
        )
        base_plots.append(fig)
    #        plot_res = (
    #            plot_func(
    #                data=query.data,
    #                x_col="year",
    #                y_col=metric,
    #                title=f"{metric} over {query.data.min().year} - {query.data.max().year}",
    #            )
    #            .configure_axis(
    #                labelFontSize=12, titleFontSize=14, labelFont="Arial", titleFont="Arial"
    #            )
    #            .configure_title(
    #                fontSize=18, font="Arial", anchor="start", color="darkblue"
    #            )
    #            .configure_legend(titleFontSize=14, labelFontSize=12)
    #            .configure_view(strokeWidth=0)
    #        )
    # base_plots =[fig]
    # base_plots.append(plot_res)

    plots = []
    grouped_metrics = group_metrics(query.metrics)
    for group, metrics in grouped_metrics.items():
        if not metrics:
            continue

        plot_name = (
            query.plots[0]["single_year"]
            if len(query.data["year"].unique().tolist()) == 1
            else query.plots[0]["multiple_years"]
        )

        g_plot = plot_multiple_metrics(
            query.data,
            x_col="year",
            metrics=metrics,
            title=f"{group} plots",
            plot_type=plot_name,
        )
        plots.append({group: g_plot})
    query.base_plots = base_plots
    query.plots = plots
    return query


def process_prompt(query: RagQuery) -> Generator[str | RagQuery, None, None]:
    
    # Сравни мегафон и мтс по долгосрочным кредитам
    
    prompt = (
        query.text_output[0]
    )
    
    for word in prompt.split():
        yield word + " "
        time.sleep(0.1)

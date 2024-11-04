import altair as alt
import random
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


# def single_bar(data:pd.DataFrame,x, y, title):
#    colors = ["#1A1A1D", "#4E4E50", "#6F2232", "#950740", "#C3073F"]
#    colors = random.choices(colors, k=len(data))
#
#    fig = px.bar(
#        data,
#        x=x,
#        y=y,
#        title=title,
#        #color_discrete_sequence=colors,
#        color=x,)
#        #colors)
#        #text="Values"           # Show values on top of bars
#
#
#    # Customize layout and text positions
#    fig.update_traces(textposition="outside")  # Place text outside bars
#    fig.update_layout(
#        title_font_size=24,
#        xaxis_title=x,
#        yaxis_title=y,
#        template="seaborn"  # Clean template for a polished look
#    )
#    return fig
def single_bar(data: pd.DataFrame, x, y, title, company_column=None):
    colors = ["#1A1A1D", "#4E4E50", "#6F2232", "#950740", "#C3073F"]
    colors = random.choices(colors, k=len(data))

    fig = px.bar(
        data,
        x=x,
        y=y,
        color=company_column,
        barmode="group",
        title=f"Изменения метрики {y}",
        color_discrete_sequence=colors,
        labels={x: y},
    )

    fig.update_layout(
        xaxis_title=x,
        yaxis_title=y,
        legend_title="Company",
    )
    return fig


# def plot_multiple_metrics(
#    data: pd.DataFrame,
#    x_col: str,
#    metrics: list,
#    plot_type: str = "line",
#    secondary_y: list = None,
#    title: str = "Multi-Metric Plot",
#    custom_colors: list = ["#1f77b4", "#ff7f0e", "#2ca02c", "#1f77b4", "#ff7f0e", "#2ca02c", "#1f77b4", "#ff7f0e", "#2ca02c"],
#    **kwargs
# ):
#    """
#    Plot multiple metrics on a specified plot type with Plotly.
#
#    Parameters:
#    - data: pd.DataFrame - Data containing the metrics to plot.
#    - x_col: str - Column to use for x-axis (e.g., Date or Category).
#    - metrics: list - List of column names in `data` to plot as metrics.
#    - plot_type: str - Type of plot to create. Options: 'line', 'bar', 'scatter'.
#    - secondary_y: list - List of metrics to plot on a secondary y-axis (default None).
#    - title: str - Title of the plot.
#    - custom_colors: list - List of colors to use for each metric.
#    - **kwargs: dict - Additional arguments for customization (e.g., size for scatter).
#
#    Returns:
#    - Plotly Figure object.
#    """
#    fig = go.Figure()
#
#    # Handle colors for each metric
#    colors = custom_colors or px.colors.qualitative.Plotly
#    if len(metrics) > len(colors):
#        colors += colors
#    #color_cycle = iter(colors)
#
#    # Define plotting based on plot type
#    for ind, metric in enumerate(metrics):
#        color = colors[ind]
#        logger.info(metric)
#        match plot_type:
#            case "line":
#                fig.add_trace(
#                    go.Scatter(
#                        x=data[x_col],
#                        y=data[metric],
#                        mode="lines+markers",
#                        name=metric,
#                        marker=dict(size=8),
#                        line=dict(width=2),
#                        yaxis="y2" if secondary_y and metric in secondary_y else "y1",
#                        line_color=color
#                    )
#                )
#            case "bar":
#                fig.add_trace(
#                    go.Bar(
#                        x=data[x_col],
#                        y=data[metric],
#                        name=metric,
#                        marker=dict(color=color),
#                        yaxis="y2" if secondary_y and metric in secondary_y else "y1"
#                    )
#                )
#            case "scatter":
#                fig.add_trace(
#                    go.Scatter(
#                        x=data[x_col],
#                        y=data[metric],
#                        mode="markers",
#                        name=metric,
#                        marker=dict(size=kwargs.get("size", 10), color=color),
#                        yaxis="y2" if secondary_y and metric in secondary_y else "y1"
#                    )
#                )
#            case _:
#                assert 1==0
#    # Update layout for secondary axis and title
#    fig.update_layout(
#    title=title,
#    xaxis_title=x_col,
#    yaxis=dict(title="Primary Y-Axis", showgrid=True),
#    yaxis2=dict(title="Secondary Y-Axis", overlaying="y", side="right") if secondary_y else None,
#    template="plotly_dark",
#    legend=dict(title="Metrics", orientation="h", x=0.5, xanchor="center", y=-0.2),
#    showlegend=True
#    )
#
#    # Enhanced hover and styling
#    fig.update_traces(hovertemplate="%{y:.2f}", selector=dict(type="scatter"))
#    fig.update_traces(textposition="outside", selector=dict(type="bar"))
#    fig.update_xaxes(showgrid=False, tickangle=45)
#    fig.update_yaxes(showgrid=True, zeroline=True)
#
#    return fig
# Set global styles for a consistent look across all charts


def plot_multiple_metrics(
    data: pd.DataFrame,
    x_col: str,
    metrics: list,
    plot_type: str = "line",
    secondary_y: list = None,
    title: str = "Multi-Metric Plot",
    custom_colors: list = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
    ],
    **kwargs,
):
    """
    Plot multiple metrics for multiple companies in a DataFrame using Plotly.

    Parameters:
    - data: pd.DataFrame - Data containing the metrics to plot, including a "comp" column for companies.
    - x_col: str - Column to use for x-axis (e.g., Date or Category).
    - metrics: list - List of column names in `data` to plot as metrics.
    - plot_type: str - Type of plot to create. Options: 'line', 'bar', 'scatter'.
    - secondary_y: list - List of metrics to plot on a secondary y-axis (default None).
    - title: str - Title of the plot.
    - custom_colors: list - List of colors to use for each company-metric combination.

    Returns:
    - Plotly Figure object.
    """
    fig = go.Figure()

    # Handle colors for each metric and company combination
    colors = custom_colors or px.colors.qualitative.Plotly
    num_colors_needed = len(data["comp"].unique()) * len(metrics)
    if num_colors_needed > len(colors):
        colors *= (num_colors_needed // len(colors)) + 1

    # Iterate through each company and add a trace per metric
    for comp_idx, (company, company_data) in enumerate(data.groupby("comp")):
        for metric_idx, metric in enumerate(metrics):
            color = colors[comp_idx * len(metrics) + metric_idx]
            yaxis = "y2" if secondary_y and metric in secondary_y else "y1"
            trace_name = f"{company} - {metric}"
            match plot_type:
                case "line":
                    fig.add_trace(
                        go.Scatter(
                            x=company_data[x_col],
                            y=company_data[metric],
                            mode="lines+markers",
                            name=trace_name,
                            marker=dict(size=8),
                            line=dict(width=2),
                            yaxis=yaxis,
                            line_color=color,
                        )
                    )
                case "bar":
                    fig.add_trace(
                        go.Bar(
                            x=company_data[x_col],
                            y=company_data[metric],
                            name=trace_name,
                            marker=dict(color=color),
                            yaxis=yaxis,
                        )
                    )
                case "scatter":
                    fig.add_trace(
                        go.Scatter(
                            x=company_data[x_col],
                            y=company_data[metric],
                            mode="markers",
                            name=trace_name,
                            marker=dict(size=kwargs.get("size", 10), color=color),
                            yaxis=yaxis,
                        )
                    )
                case _:
                    raise ValueError(f"Unsupported plot type: {plot_type}")

    # Update layout for secondary axis, title, and other features
    fig.update_layout(
        title=title,
        xaxis_title=x_col,
        yaxis=dict(title="Primary Y-Axis", showgrid=True),
        yaxis2=dict(title="Secondary Y-Axis", overlaying="y", side="right")
        if secondary_y
        else None,
        template="plotly_dark",
        legend=dict(title="Metrics", orientation="h", x=0.5, xanchor="center", y=-0.2),
        showlegend=True,
    )

    # Enhanced hover and styling
    fig.update_traces(hovertemplate="%{y:.2f}", selector=dict(type="scatter"))
    fig.update_traces(textposition="outside", selector=dict(type="bar"))
    fig.update_xaxes(showgrid=False, tickangle=45)
    fig.update_yaxes(showgrid=True, zeroline=True)

    return fig


def create_bar_plot(data: pd.DataFrame, x_col, y_col, title="Bar Plot"):
    """Create an enhanced interactive bar plot."""
    chart = (
        alt.Chart(data)
        .mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3)
        .encode(
            x=alt.X(x_col, title=x_col, sort="-y"),
            y=alt.Y(y_col, title=y_col),
            color=alt.Color(x_col, legend=None),  # Custom color based on category
            tooltip=[x_col, y_col],
        )
        .interactive()
    )  # Make it interactive
    return chart


def create_scatter_plot(
    data: pd.DataFrame, x_col, y_col, size_col=None, title="Scatter Plot"
):
    """Create an enhanced interactive scatter plot with zoom and pan."""
    chart = (
        alt.Chart(data)
        .mark_circle()
        .encode(
            x=alt.X(x_col, title=x_col),
            y=alt.Y(y_col, title=y_col),
            size=alt.Size(size_col, legend=None) if size_col else alt.value(60),
            color=alt.Color(x_col, legend=None),
            tooltip=[x_col, y_col],
        )
        .interactive()
    )  # Enable zooming and panning
    return chart


def create_pie_chart(data: pd.DataFrame, category_col, value_col, title="Pie Chart"):
    """Create an interactive pie chart."""
    # Calculate percentage
    data["percentage"] = data[value_col] / data[value_col].sum() * 100
    chart = (
        alt.Chart(data)
        .mark_arc(innerRadius=50)
        .encode(
            theta=alt.Theta(field="percentage", type="quantitative"),
            color=alt.Color(field=category_col, type="nominal", legend=None),
            tooltip=[
                alt.Tooltip(category_col, title="Category"),
                alt.Tooltip(value_col, title="Value"),
                alt.Tooltip("percentage:Q", format=".1f", title="Percentage"),
            ],
        )
        .properties(title=title)
        .configure_legend(orient="right")
    )
    return chart


def create_line_plot(data: pd.DataFrame, x_col, y_col, title="Line Plot"):
    """Create an interactive line plot with hover tooltip."""
    line = (
        alt.Chart(data)
        .mark_line()
        .encode(
            x=alt.X(x_col, title=x_col),
            y=alt.Y(y_col, title=y_col),
            color=alt.Color(x_col, legend=None),
        )
    )
    points = line.mark_point(size=100).encode(tooltip=[x_col, y_col])
    chart = (line + points).interactive()  # Enable hover tooltip on points
    chart = (
        chart.properties(title=title)
        .configure_axis(grid=True)
        .configure_view(strokeWidth=0)
    )
    return chart


# Display the plots
# plot_debt_ratios(df) & plot_interest_coverage(df) & plot_cash_flow(df) & plot_profit_margin(df) & plot_RoE_RoA(df)

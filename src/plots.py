from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def plot_regime_probs(probs: pd.DataFrame) -> go.Figure:
    fig = px.area(probs, x=probs.index, y=probs.columns, title="Regime probabilities")
    fig.update_layout(legend_title="Regime", yaxis_tickformat=".0%")
    return fig


def plot_timeline(state: pd.Series) -> go.Figure:
    df = state.to_frame("regime")
    mapper = {k: i for i, k in enumerate(["Goldilocks", "Reflation", "Slowdown", "Stagflation"])}
    df["code"] = df["regime"].map(mapper)
    fig = px.scatter(df, x=df.index, y="code", color="regime", title="Regime timeline")
    fig.update_yaxes(tickvals=list(mapper.values()), ticktext=list(mapper.keys()))
    return fig


def plot_curve_spread(curve_df: pd.DataFrame) -> go.Figure:
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=curve_df.index, y=curve_df["DGS10"], name="US 10Y"), secondary_y=False)
    fig.add_trace(go.Scatter(x=curve_df.index, y=curve_df["DGS2"], name="US 2Y"), secondary_y=False)
    fig.add_trace(go.Scatter(x=curve_df.index, y=curve_df["Slope"], name="10s-2s"), secondary_y=True)
    fig.update_layout(title="US Curve & Slope")
    return fig

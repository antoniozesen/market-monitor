from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def curve_schema(macro: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=macro.index if not macro.empty else pd.DatetimeIndex([]), columns=["2Y", "10Y", "Slope"], dtype=float)
    if macro.empty:
        return out
    c2 = [c for c in macro.columns if "DGS2" in c]
    c10 = [c for c in macro.columns if "DGS10" in c]
    if c2:
        out["2Y"] = macro[c2].mean(axis=1)
    if c10:
        out["10Y"] = macro[c10].mean(axis=1)
    out["Slope"] = out["10Y"] - out["2Y"]
    return out


def fig_curve(curve: pd.DataFrame) -> go.Figure:
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    if "2Y" in curve.columns:
        fig.add_trace(go.Scatter(x=curve.index, y=curve["2Y"], name="2Y"), secondary_y=False)
    if "10Y" in curve.columns:
        fig.add_trace(go.Scatter(x=curve.index, y=curve["10Y"], name="10Y"), secondary_y=False)
    if "Slope" in curve.columns:
        fig.add_trace(go.Scatter(x=curve.index, y=curve["Slope"], name="Slope"), secondary_y=True)
    fig.update_layout(title="US Curve (2Y/10Y) and Slope")
    return fig


def fig_heatmap(df: pd.DataFrame, title: str):
    if df.empty:
        return px.imshow(np.zeros((1, 1)), title=title)
    return px.imshow(df, aspect="auto", title=title)

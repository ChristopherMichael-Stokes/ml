import pandas as pd
import plotly
import plotly.express as px


def col_histogram(df: pd.DataFrame, col: str, **kwargs) -> plotly.graph_objs._figure.Figure:
    fig = px.histogram(df, x=col, title=f"Distribution of {col}", **kwargs)
    return fig

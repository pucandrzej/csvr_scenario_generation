import plotly.graph_objects as go


def add_curve(fig, x, y, name, color):
    """
    Add a curve trace to a Plotly figure.
    """
    fig.add_trace(
        go.Scatter(x=x, y=y, mode="lines+markers", name=name, line=dict(color=color))
    )

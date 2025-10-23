import numpy as np
import plotly.graph_objects as go
from core import rgb_to_hex

def plot_macadam_plotly(
    Lab_ref,
    rgb,
    hue_deg=None,
    chroma=None,
    title="Rösch–MacAdam Colour Solid (CIELAB, D65)",
    max_points=60000,
    marker_size=5,
    opacity=1.0
):
    L = Lab_ref[:, 0]
    a = Lab_ref[:, 1]
    b = Lab_ref[:, 2]
    N = Lab_ref.shape[0]

    if N > max_points:
        idx = np.linspace(0, N - 1, max_points).astype(int)
        a, b, L, rgb = a[idx], b[idx], L[idx], rgb[idx]
        if hue_deg is not None:
            hue_deg = hue_deg[idx]
        if chroma is not None:
            chroma = chroma[idx]

    colors_hex = rgb_to_hex(rgb)

    if hue_deg is not None and chroma is not None:
        hover = (
            'a*: %{x:.2f}<br>b*: %{y:.2f}<br>L*: %{z:.2f}'
            '<br>h*: %{customdata[0]:.2f}°<br>C*: %{customdata[1]:.2f}'
            '<extra></extra>'
        )
        custom = np.column_stack([hue_deg, chroma])
    else:
        hover = 'a*: %{x:.2f}<br>b*: %{y:.2f}<br>L*: %{z:.2f}<extra></extra>'
        custom = None

    factor = 1.0
    fig = go.Figure(
        data=go.Scatter3d(
            x=a * factor, y=b * factor, z=L * factor,
            mode='markers',
            marker=dict(size=marker_size, color=colors_hex, opacity=opacity),
            hovertemplate=hover,
            customdata=custom
        )
    )

    fig.update_layout(
        scene=dict(
            xaxis_title='a*',
            yaxis_title='b*',
            zaxis_title='L*',
            aspectmode='cube' # choose e.g. 'data' for a compressed view in Z axis
        ),
        title=title,
        template='plotly_white',
        margin=dict(l=0, r=0, t=40, b=0),
        showlegend=False
    )

    return fig
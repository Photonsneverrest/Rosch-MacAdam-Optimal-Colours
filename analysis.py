import numpy as np

_CIELAB_EPSILON = 216 / 24389  # ≈ 0.008856
_CIELAB_KAPPA = 24389 / 27     # ≈ 903.3

def angular_distance_deg(a, b):
    """Smallest absolute angular difference a↔b in degrees (0..180]."""
    return np.abs((a - b + 180.0) % 360.0 - 180.0)

def max_chroma_at_hue(Lab_ref, hue_deg, chroma, target_h, tol_deg=0.5):
    mask = angular_distance_deg(hue_deg, target_h) <= tol_deg
    if not np.any(mask):
        return None
    idxs = np.nonzero(mask)[0]
    i_best = idxs[np.argmax(chroma[mask])]
    L, a, b = Lab_ref[i_best]
    return dict(idx=int(i_best), L=float(L), a=float(a), b=float(b),
                C=float(chroma[i_best]), h=float(hue_deg[i_best]))

def chroma_envelope_vs_L_for_hue(Lab_ref, hue_deg, chroma, target_h,
                                  tol_deg=1.0, L_min=0.0, L_max=100.0, L_step=1.0):
    L_vals = Lab_ref[:, 0]
    bins = np.arange(L_min, L_max + L_step, L_step)
    L_centres, C_max_list, a_list, b_list = [], [], [], []
    hue_mask = angular_distance_deg(hue_deg, target_h) <= tol_deg
    if not np.any(hue_mask):
        return np.array([]), np.array([]), np.array([]), np.array([])
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = hue_mask & (L_vals >= lo) & (L_vals < hi)
        if np.any(m):
            idxs = np.nonzero(m)[0]
            imax = idxs[np.argmax(chroma[m])]
            L_centres.append(0.5 * (lo + hi))
            C_max_list.append(chroma[imax])
            a_list.append(Lab_ref[imax, 1])
            b_list.append(Lab_ref[imax, 2])
    return (np.asarray(L_centres), np.asarray(C_max_list),
            np.asarray(a_list), np.asarray(b_list))

def Lstar_to_Yrel(Lstar):
    L = np.asarray(Lstar, dtype=float)
    fy = (L + 16.0) / 116.0
    Yrel = np.where(L > (_CIELAB_KAPPA * _CIELAB_EPSILON), fy**3, L / _CIELAB_KAPPA)
    return Yrel

def max_chroma_stats_at_hue(Lab_ref, hue_deg, chroma, target_h, tol_deg=0.75):
    mask = angular_distance_deg(hue_deg, float(target_h)) <= float(tol_deg)
    if not np.any(mask):
        return None
    idxs = np.nonzero(mask)[0]
    chroma_scaled = 10.0 * chroma
    local_chroma = chroma_scaled[mask]
    best_local = idxs[np.argmax(local_chroma)]
    Lab_ref_scaled = 10.0 * Lab_ref
    L, a, b = Lab_ref_scaled[best_local]
    C = float(chroma_scaled[best_local])
    h = float(hue_deg[best_local])
    Y_rel = float(Lstar_to_Yrel(L))
    return dict(idx=int(best_local), h=h, C=C, L=float(L), a=float(a), b=float(b), Y_rel=Y_rel)

import plotly.graph_objects as go
import numpy as np

def add_hue_ridge_to_fig(fig, L_centres, a_vals, b_vals,
                          scale=10.0, name='Max-C* ridge at hue',
                          color='black'):
    if L_centres.size == 0:
        return fig
    fig.add_trace(go.Scatter3d(
        x=a_vals * scale,
        y=b_vals * scale,
        z=L_centres * scale,
        mode='lines+markers',
        name=name,
        line=dict(color=color, width=4),
        marker=dict(size=3, color=color),
        hovertemplate='L*: %{z:.2f}<br>a*: %{x:.2f}<br>b*: %{y:.2f}<extra></extra>'
    ))
    return fig
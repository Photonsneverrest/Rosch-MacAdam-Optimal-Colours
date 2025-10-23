import numpy as np
from scipy.spatial import ConvexHull
import plotly.graph_objects as go
import colour # type: ignore

def macadam_mesh_in_Lab_from_XYZ_hull(XYZ_rel, Lab_ref, scale=1.0):
    """
    Build a triangular mesh of the Rösch–MacAdam outer surface:
    - Compute the convex hull in XYZ (convex domain).
    - Map vertices to CIELAB for display.
    Returns (V, F) where V are Lab vertices (scaled for your plot) and F are triangle indices.
    """
    hull = ConvexHull(XYZ_rel)
    simplices = hull.simplices
    unique_idx = np.unique(simplices.ravel())
    old_to_new = {old: new for new, old in enumerate(unique_idx)}
    V_lab = Lab_ref[unique_idx]
    F = np.vectorize(old_to_new.get)(simplices)
    V_plot = np.column_stack([
        V_lab[:, 1] * scale,  # a*
        V_lab[:, 2] * scale,  # b*
        V_lab[:, 0] * scale   # L*
    ])
    return V_plot, F

def add_mesh3d(fig, V_plot, F, color='rgba(80,80,80,0.25)', name='Macadam Mesh',
               showlegend=True, lighting=True):
    i, j, k = F[:, 0], F[:, 1], F[:, 2]
    lighting_dict = dict(ambient=0.3, diffuse=0.8, roughness=0.9,
                         fresnel=0.1, specular=0.2) if lighting else None
    fig.add_trace(go.Mesh3d(
        x=V_plot[:, 0], y=V_plot[:, 1], z=V_plot[:, 2],
        i=i, j=j, k=k,
        color=color,
        opacity=float(color.split(',')[-1].rstrip(')')) if 'rgba' in color else 0.25,
        name=name,
        showlegend=showlegend,
        lighting=lighting_dict
    ))
    return fig

def ebu_colourspace():
    names = list(colour.RGB_COLOURSPACES.keys())
    candidates = [
        'EBU Tech. 3213-E',
        'ITU-R BT.470 - 625-Line',
        'PAL/SECAM'
    ]
    for n in candidates:
        if n in colour.RGB_COLOURSPACES:
            return colour.RGB_COLOURSPACES[n], n
    raise KeyError("Could not find an EBU RGB colourspace.")

def _face_grid(n=33):
    t = np.linspace(0.0, 1.0, n)
    return np.meshgrid(t, t, indexing='ij')

def _grid_tris(n):
    tris = []
    for r in range(n - 1):
        for c in range(n - 1):
            p0 = r * n + c
            p1 = p0 + 1
            p2 = p0 + n
            p3 = p2 + 1
            tris.append((p0, p1, p3))
            tris.append((p0, p3, p2))
    tris = np.asarray(tris, int)
    return tris[:, 0], tris[:, 1], tris[:, 2]

def _rgb_linear_to_XYZ_with_cs(RGB, cs):
    try:
        return colour.RGB_to_XYZ(RGB, cs, chromatic_adaptation_transform=None)
    except TypeError:
        return colour.RGB_to_XYZ(
            RGB,
            illuminant_RGB=cs.whitepoint,
            illuminant_XYZ=cs.whitepoint,
            matrix_RGB_to_XYZ=cs.matrix_RGB_to_XYZ,
            chromatic_adaptation_transform=None
        )

def build_ebu_lab_mesh(n=33, scale=1.0):
    cs, cs_name = ebu_colourspace()
    V_list = []
    F_list = []
    offset = 0

    def add_face(R_lin, G_lin, B_lin):
        nonlocal V_list, F_list, offset
        RGB = np.stack([R_lin.ravel(), G_lin.ravel(), B_lin.ravel()], axis=-1)
        XYZ = _rgb_linear_to_XYZ_with_cs(RGB, cs)
        with colour.domain_range_scale('reference'):
            Lab = colour.XYZ_to_Lab(XYZ, illuminant=cs.whitepoint)
        V = np.column_stack([Lab[:, 1] * scale, Lab[:, 2] * scale, Lab[:, 0] * scale])
        V_list.append(V)
        i, j, k = _grid_tris(R_lin.shape[0])
        F_list.append(np.column_stack([i + offset, j + offset, k + offset]))
        offset += V.shape[0]

    U, V = _face_grid(n)
    add_face(np.zeros_like(U), U, V)
    add_face(np.ones_like(U), U, V)
    add_face(U, np.zeros_like(U), V)
    add_face(U, np.ones_like(U), V)
    add_face(U, V, np.zeros_like(U))
    add_face(U, V, np.ones_like(U))

    V_plot = np.vstack(V_list)
    F = np.vstack(F_list)
    return V_plot, F, cs_name

def add_ebu_mesh(fig, n=41, color='rgba(0,120,255,0.30)', name_prefix='RGB EBU'):
    V_plot, F, cs_name = build_ebu_lab_mesh(n=n, scale=1.0)
    i, j, k = F[:, 0], F[:, 1], F[:, 2]
    fig.add_trace(go.Mesh3d(
        x=V_plot[:, 0], y=V_plot[:, 1], z=V_plot[:, 2],
        i=i, j=j, k=k,
        color=color,
        name=f'{name_prefix} ({cs_name})',
        showlegend=True,
        lighting=dict(ambient=0.5, diffuse=0.6, roughness=1.0)
    ))
    return fig
#%%
import numpy as np
import plotly.graph_objects as go
import colour

def build_rosch_macadam_XYZ(
    cmfs_name='CIE 1931 2 Degree Standard Observer',
    illuminant_name='D65',
    bins=None,
    point_order='Pulse Wave Width',   # <— ensure we sweep widths
    filter_jagged_points=False        # <— keep all widths, then you can filter later
):
    """
    Returns:
      XYZ_rel   : (N, 3) XYZ with Y_white = 1 (relative luminance factor)
      Lab_ref   : (N, 3) CIELAB (L* 0..100, a*, b* in conventional units)  [computed in 'reference']
      rgb_phys  : (N, 3) sRGB (gamma) from XYZ_rel (physically scaled; often dark)
      rgb_bright: (N, 3) sRGB (gamma) from exposure-scaled XYZ_rel (for display)
      hue_deg   : (N,)   CIELAB hue angle in degrees
      chroma    : (N,)   CIELAB chroma C*ab
      debug     : dict   helpful ranges (Yw, Y ranges, scale states, pulses config)
    """
    # Work in '1' domain for spectral computations & sRGB.
    colour.set_domain_range_scale('1')

    # Clear caches so we don't reuse an old, narrow-pulse outer surface.
    try:
        colour.utilities.CACHE_REGISTRY.clear()
    except Exception:
        pass

    # Align CMFs & illuminant to Colour's outer-surface shape (360–780 nm, 5 nm).
    shape = colour.volume.spectrum.SPECTRAL_SHAPE_OUTER_SURFACE_XYZ
    cmfs = colour.MSDS_CMFS[cmfs_name].copy().align(shape)
    illuminant = colour.SDS_ILLUMINANTS[illuminant_name].copy().align(shape)

    # Choose bins equal to the number of spectral samples if not given.
    if bins is None:
        bins = cmfs.shape.wavelengths.size  # typically 81 for 360–780 with 5 nm

    # Build Rösch–MacAdam outer surface (XYZ).
    # NOTE: The function accepts point_order, filter_jagged_points, and passes **kwargs to
    # its internals; current Colour versions accept 'bins' to control pulse discretisation.
    # Docs: colour.volume.solid_RoschMacAdam (outer surface from pulse waves).
    XYZ_outer = colour.volume.solid_RoschMacAdam(
        cmfs=cmfs,
        illuminant=illuminant,
        point_order=point_order,
        filter_jagged_points=filter_jagged_points,
        bins=bins,  # <— critical so we cover widths from 1..bins
    )

    # Perfect diffuser white under same illuminant/CMFs (in '1' domain → Yw ≈ 1).
    R_white = colour.SpectralDistribution(
        data=np.ones_like(illuminant.values),
        domain=illuminant.wavelengths
    )
    Xw, Yw, Zw = colour.sd_to_XYZ(R_white, cmfs=cmfs, illuminant=illuminant)

    # Relative XYZ (white Y = 1).
    XYZ_rel = XYZ_outer / Yw

    # Convert to Lab in conventional units (L* 0..100) by temporarily switching to 'reference'.
    D65_xy = colour.CCS_ILLUMINANTS['CIE 1931 2 Degree Standard Observer']['D65']
    with colour.domain_range_scale('reference'):
        Lab_ref = colour.XYZ_to_Lab(XYZ_rel, illuminant=D65_xy)

    # Hue & chroma from Lab (conventional units).
    a = Lab_ref[:, 1]
    b = Lab_ref[:, 2]
    chroma = np.hypot(a, b)
    hue_deg = (np.degrees(np.arctan2(b, a)) % 360.0)

    # Marker colours from sRGB (gamma); physically accurate but often dark.
    rgb_phys = np.clip(colour.XYZ_to_sRGB(XYZ_rel, illuminant=D65_xy), 0.0, 1.0)

    # Brighten for display only: map the 99th percentile of Y to ~1.
    Y_rel = XYZ_rel[:, 1]
    Y_nonzero = Y_rel[Y_rel > 0]
    gain = np.percentile(Y_nonzero, 99.0) if Y_nonzero.size else 1.0
    gain = gain if gain > 0 else 1.0
    XYZ_disp = XYZ_rel / gain
    rgb_bright = np.clip(
        colour.XYZ_to_sRGB(np.maximum(XYZ_disp, 0), illuminant=D65_xy),
        0.0, 1.0
    )

    debug = {
        'domain_global': colour.get_domain_range_scale(),
        'bins': bins,
        'point_order': point_order,
        'filter_jagged_points': filter_jagged_points,
        'Yw': float(Yw),
        'XYZ_outer_Y_minmax': (float(XYZ_outer[:, 1].min()), float(XYZ_outer[:, 1].max())),
        'XYZ_rel_Y_minmax': (float(XYZ_rel[:, 1].min()), float(XYZ_rel[:, 1].max())),
    }
    return XYZ_rel, Lab_ref, rgb_phys, rgb_bright, hue_deg, chroma, debug


def rgb_to_hex(rgb):
    rgb8 = (np.clip(rgb, 0, 1) * 255 + 0.5).astype(np.uint8)
    return [f'#{r:02X}{g:02X}{b:02X}' for r, g, b in rgb8]


def plot_macadam_plotly(Lab_ref,
                        rgb,
                        hue_deg=None,
                        chroma=None,
                        title="Rösch–MacAdam Colour Solid (CIELAB, D65)",
                        max_points=60000,
                        marker_size=5,
                        opacity=1.0):
    L = Lab_ref[:, 0]; a = Lab_ref[:, 1]; b = Lab_ref[:, 2]
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
        hover = ('a*: %{x:.2f}<br>b*: %{y:.2f}<br>L*: %{z:.2f}'
                 '<br>h*: %{customdata[0]:.2f}°<br>C*: %{customdata[1]:.2f}'
                 '<extra></extra>')
        custom = np.column_stack([hue_deg, chroma])
    else:
        hover = 'a*: %{x:.2f}<br>b*: %{y:.2f}<br>L*: %{z:.2f}<extra></extra>'
        custom = None

    factor = 10.0
    fig = go.Figure(
        data=go.Scatter3d(
            x=a*factor, y=b*factor, z=L*factor,
            mode='markers',
            marker=dict(size=marker_size, color=colors_hex, opacity=opacity),
            hovertemplate=hover,
            customdata=custom
        )
    )
    fig.update_layout(
        scene=dict(xaxis_title='a*', yaxis_title='b*', zaxis_title='L*', aspectmode='data'),
        title=title, template='plotly_white',
        margin=dict(l=0, r=0, t=40, b=0), showlegend=False
    )
    return fig


if __name__ == "__main__":
    # Build with explicit full-width sweep: bins = number of samples (≈81).
    XYZ_rel, Lab_ref, rgb_phys, rgb_bright, hue_deg, chroma, debug = build_rosch_macadam_XYZ(
        cmfs_name='CIE 1931 2 Degree Standard Observer',
        illuminant_name='D65',
        bins=None,                         # will auto set to 81
        point_order='Pulse Wave Width',
        filter_jagged_points=False
    )

    fig = plot_macadam_plotly(
        Lab_ref,
        rgb_bright,      # use brightened colours for visibility
        hue_deg=hue_deg,
        chroma=chroma,
        title="Rösch–MacAdam Colour Solid (CIELAB, D65) — Plotly",
        max_points=80000,
        marker_size=5,
        opacity=1.0
    )
    try:
        fig.show()
    except Exception as e:
        import webbrowser, tempfile, pathlib
        print(f"Plotly inline show failed: {e}\nSaving to HTML and opening in browser.")
        out_path = pathlib.Path(tempfile.gettempdir()) / 'rosch_macadam_colour_solid.html'
        fig.write_html(str(out_path), include_plotlyjs='cdn')
        webbrowser.open(out_path.as_uri())
# %%
import numpy as np
import plotly.graph_objects as go
from scipy.spatial import ConvexHull
import colour

# -----------------------------
# A) Triangulated mesh of optimal colour solid
# -----------------------------
def macadam_mesh_in_Lab_from_XYZ_hull(XYZ_rel, Lab_ref, scale=10.0):
    """
    Build a triangular mesh of the Rösch–MacAdam outer surface:
    - Compute the convex hull in XYZ (convex domain).
    - Map vertices to CIELAB for display.
    Returns (V, F) where V are Lab vertices (scaled for your plot) and F are triangle indices.
    """
    hull = ConvexHull(XYZ_rel)
    simplices = hull.simplices  # (M, 3) triangles indexing the original point set
    # Compress to unique vertex list for a smaller mesh.
    unique_idx = np.unique(simplices.ravel())
    old_to_new = {old: new for new, old in enumerate(unique_idx)}

    V_lab = Lab_ref[unique_idx]  # (K, 3)
    F = np.vectorize(old_to_new.get)(simplices)  # (M, 3)

    # Scale the Lab coordinates the same way you scaled the scatter (a*, b*, L*)
    V_plot = np.column_stack([V_lab[:, 1] * scale,  # a*
                              V_lab[:, 2] * scale,  # b*
                              V_lab[:, 0] * scale]) # L*
    return V_plot, F


def add_mesh3d(fig, V_plot, F, color='rgba(80,80,80,0.25)', name='Macadam Mesh',
               showlegend=True, lighting=True):
    i, j, k = F[:, 0], F[:, 1], F[:, 2]
    lighting_dict = dict(ambient=0.3, diffuse=0.8, roughness=0.9, fresnel=0.1, specular=0.2) if lighting else None
    fig.add_trace(go.Mesh3d(
        x=V_plot[:, 0], y=V_plot[:, 1], z=V_plot[:, 2],
        i=i, j=j, k=k,
        color=color, opacity=float(color.split(',')[-1].rstrip(')')) if 'rgba' in color else 0.25,
        name=name, showlegend=showlegend, lighting=lighting_dict
    ))
    return fig


# -----------------------------
# B) Max chroma at a given hue (and envelope vs L*)
# -----------------------------
def angular_distance_deg(a, b):
    d = np.abs((a - b + 180.0) % 360.0 - 180.0)
    return d

def max_chroma_at_hue(Lab_ref, hue_deg, chroma, target_h, tol_deg=0.5):
    """
    Returns the single point on the surface (within tol) that has maximum chroma:
    dict with keys: idx, L, a, b, C, h
    """
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
    """
    For each L* bin, pick the point (within hue tol) with maximum chroma.
    Returns arrays (L_centres, C_max, a_vals, b_vals) for plotting a ridge line.
    """
    L_vals = Lab_ref[:, 0]
    bins = np.arange(L_min, L_max + L_step, L_step)
    L_centres, C_max_list, a_list, b_list = [], [], [], []

    hue_mask = angular_distance_deg(hue_deg, target_h) <= tol_deg
    if not np.any(hue_mask):
        return (np.array([]), np.array([]), np.array([]), np.array([]))

    for lo, hi in zip(bins[:-1], bins[1:]):
        m = hue_mask & (L_vals >= lo) & (L_vals < hi)
        if np.any(m):
            idxs = np.nonzero(m)[0]
            imax = idxs[np.argmax(chroma[m])]
            L_centres.append(0.5*(lo+hi))
            C_max_list.append(chroma[imax])
            a_list.append(Lab_ref[imax, 1])
            b_list.append(Lab_ref[imax, 2])

    return (np.asarray(L_centres), np.asarray(C_max_list),
            np.asarray(a_list), np.asarray(b_list))


def add_hue_ridge_to_fig(fig, L_centres, a_vals, b_vals,
                         scale=10.0, name='Max-C* ridge at hue',
                         color='black'):
    if L_centres.size == 0:
        return fig
    fig.add_trace(go.Scatter3d(
        x=a_vals*scale, y=b_vals*scale, z=L_centres*scale,
        mode='lines+markers', name=name,
        line=dict(color=color, width=4),
        marker=dict(size=3, color=color),
        hovertemplate='L*: %{z:.2f}<br>a*: %{x:.2f}<br>b*: %{y:.2f}<extra></extra>'
    ))
    return fig


# -----------------------------
# C) RGB_{EBU} gamut surface in CIELAB (D65)
# -----------------------------
def ebu_colourspace():
    # Robust lookup for the EBU Tech. 3213-E colourspace in 'colour'
    names = list(colour.RGB_COLOURSPACES.keys())
    candidates = [
        'EBU Tech. 3213-E',
        'ITU-R BT.470 - 625-Line',  # PAL/SECAM (close/alias depending on version)
        'PAL/SECAM'
    ]
    for n in candidates:
        if n in colour.RGB_COLOURSPACES:
            return colour.RGB_COLOURSPACES[n], n
    raise KeyError(f"Could not find an EBU RGB colourspace in installed 'colour'. "
                   f"Available spaces include e.g.: {names[:10]} ...")


def _face_grid(n=33):
    t = np.linspace(0.0, 1.0, n)
    return np.meshgrid(t, t, indexing='ij')


def _grid_tris(n):
    tris = []
    for r in range(n-1):
        for c in range(n-1):
            p0 = r*n + c
            p1 = p0 + 1
            p2 = p0 + n
            p3 = p2 + 1
            tris.append((p0, p1, p3))
            tris.append((p0, p3, p2))
    tris = np.asarray(tris, int)
    return tris[:,0], tris[:,1], tris[:,2]


def _rgb_linear_to_XYZ_with_cs(RGB, cs):
    """
    Convert LINEAR RGB (in the given colourspace) to XYZ using the installed
    'colour' API version safely.
    """
    try:
        # Newer API: RGB_to_XYZ(RGB, colourspace, ...)
        return colour.RGB_to_XYZ(RGB, cs, chromatic_adaptation_transform=None)
    except TypeError:
        # Older API: RGB_to_XYZ(RGB, illuminant_RGB, illuminant_XYZ, matrix, ...)
        return colour.RGB_to_XYZ(
            RGB,
            illuminant_RGB=cs.whitepoint,
            illuminant_XYZ=cs.whitepoint,
            matrix_RGB_to_XYZ=cs.matrix_RGB_to_XYZ,
            chromatic_adaptation_transform=None
        )


def build_ebu_lab_mesh(n=33, scale=10.0):
    """
    Build a mesh approximating the EBU RGB gamut boundary by sampling the 6 faces
    of the *linear-light* RGB cube and mapping to CIELAB (D65).
    Returns (V_plot, F, cs_name).
    """
    cs, cs_name = ebu_colourspace()

    V_list = []
    F_list = []
    offset = 0

    def add_face(R_lin, G_lin, B_lin):
        nonlocal V_list, F_list, offset
        RGB = np.stack([R_lin.ravel(), G_lin.ravel(), B_lin.ravel()], axis=-1)  # linear-light
        XYZ = _rgb_linear_to_XYZ_with_cs(RGB, cs)
        with colour.domain_range_scale('reference'):
            Lab = colour.XYZ_to_Lab(XYZ, illuminant=cs.whitepoint)
        V = np.column_stack([Lab[:,1]*scale, Lab[:,2]*scale, Lab[:,0]*scale])
        V_list.append(V)
        i, j, k = _grid_tris(R_lin.shape[0])
        F_list.append(np.column_stack([i+offset, j+offset, k+offset]))
        offset += V.shape[0]

    # Build all 6 faces on a regular grid (linear light).
    U, V = _face_grid(n)
    add_face(np.zeros_like(U), U, V)  # R=0
    add_face(np.ones_like(U),  U, V)  # R=1
    add_face(U, np.zeros_like(U), V)  # G=0
    add_face(U, np.ones_like(U),  V)  # G=1
    add_face(U, V, np.zeros_like(U))  # B=0
    add_face(U, V, np.ones_like(U))   # B=1
    V_plot = np.vstack(V_list)
    F = np.vstack(F_list)
    return V_plot, F, cs_name


def add_ebu_mesh(fig, n=41, color='rgba(0,120,255,0.30)', name_prefix='RGB EBU'):
    V_plot, F, cs_name = build_ebu_lab_mesh(n=n, scale=1.0)
    i, j, k = F[:, 0], F[:, 1], F[:, 2]
    fig.add_trace(go.Mesh3d(
        x=V_plot[:,0], y=V_plot[:,1], z=V_plot[:,2],
        i=i, j=j, k=k,
        color=color, name=f'{name_prefix} ({cs_name})',
        showlegend=True,
        lighting=dict(ambient=0.5, diffuse=0.6, roughness=1.0)
    ))
    return fig


# ---- Constants for CIELAB inverse ----
_CIELAB_EPSILON = 216 / 24389     # ≈ 0.008856
_CIELAB_KAPPA   = 24389 / 27      # ≈ 903.3

def angular_distance_deg(a, b):
    """Smallest absolute angular difference a↔b in degrees (0..180]."""
    return np.abs((a - b + 180.0) % 360.0 - 180.0)

def Lstar_to_Yrel(Lstar):
    """
    Convert CIELAB L* (0..100) to relative luminance Y/Yn (0..1+) per CIE 1976.
    Uses the piecewise inverse:
      If L* > 8:    Y/Yn = ((L* + 16) / 116) ** 3
      Else:         Y/Yn = L* / κ
    where κ = 24389/27, ε = 216/24389 (threshold corresponds to L* = κ ε = 8).
    """
    L = np.asarray(Lstar, dtype=float)
    fy = (L + 16.0) / 116.0
    Yrel = np.where(L > (_CIELAB_KAPPA * _CIELAB_EPSILON), fy**3, L / _CIELAB_KAPPA)
    return Yrel

def max_chroma_stats_at_hue(Lab_ref, hue_deg, chroma,
                            target_h, tol_deg=0.75):
    """
    Find the *single* surface point with maximum C* at the specified hue (± tol).

    Parameters
    ----------
    Lab_ref  : (N,3) array of [L*, a*, b*] in *reference* scaling (L* 0..100).
    hue_deg  : (N,) array of hue angles (0..360).
    chroma   : (N,) array of C*ab.
    target_h : float, target hue in degrees.
    tol_deg  : float, angular tolerance (deg) to gather candidates.

    Returns
    -------
    result : dict or None
        dict with keys:
          'idx', 'h', 'C', 'L', 'a', 'b', 'Y_rel'
        or None if no points within tolerance.
    """
    # gather candidates near hue
    mask = angular_distance_deg(hue_deg, float(target_h)) <= float(tol_deg)
    if not np.any(mask):
        return None
    idxs = np.nonzero(mask)[0]
    # pick max chroma among candidates
    chroma_scaled = 10.0*chroma  # scale to conventional units
    local_chroma = chroma_scaled[mask]
    best_local = idxs[np.argmax(local_chroma)]
    Lab_ref_scaled = 10.0*Lab_ref  # scale to conventional units
    L, a, b = Lab_ref_scaled[best_local]
    C = float(chroma_scaled[best_local])
    h = float(hue_deg[best_local])
    Y_rel = float(Lstar_to_Yrel(L))

    return dict(
        idx=int(best_local),
        h=h,
        C=C,
        L=float(L),
        a=float(a),
        b=float(b),
        Y_rel=Y_rel
    )
# -----------------------------
# D) Example usage in your __main__
# -----------------------------
if __name__ == "__main__":
    XYZ_rel, Lab_ref, rgb_phys, rgb_bright, hue_deg, chroma, debug = build_rosch_macadam_XYZ(
        cmfs_name='CIE 1931 2 Degree Standard Observer',
        illuminant_name='D65',
        bins=None,
        point_order='Pulse Wave Width',
        filter_jagged_points=False
    )

    fig = plot_macadam_plotly(
        Lab_ref,
        rgb_bright,
        hue_deg=hue_deg,
        chroma=chroma,
        title="Rösch–MacAdam Colour Solid (CIELAB, D65) — Plotly",
        max_points=80000,
        marker_size=5,
        opacity=1.0
    )

    # A) Add triangulated mesh of the optimal-colour outer surface
    V_plot, F = macadam_mesh_in_Lab_from_XYZ_hull(XYZ_rel, Lab_ref, scale=10.0)
    fig = add_mesh3d(fig, V_plot, F, color='rgba(80,80,80,0.18)', name='Optimal Colours Mesh')

    # B) Query and plot "max chroma at a specific hue"
    target_hue = 40.0   # degrees; change as you like
    tol_deg = 0.75

    peak = max_chroma_at_hue(Lab_ref, hue_deg, chroma, target_hue, tol_deg=tol_deg)
    if peak is not None:
        # Mark the single best point
        fig.add_trace(go.Scatter3d(
            x=[peak['a']*10.0], y=[peak['b']*10.0], z=[peak['L']*10.0],
            mode='markers', marker=dict(size=7, color='crimson', symbol='x'),
            name=f'Max C* at h={target_hue:.1f}° (C*={peak["C"]:.1f}, L*={peak["L"]:.1f})',
            hovertemplate='h*: ' + f'{peak["h"]:.2f}°' +
                          '<br>C*: %{customdata[0]:.2f}<br>L*: %{z:.2f}<extra></extra>',
            customdata=np.array([[peak['C']]])
        ))

        # Ridge: max chroma vs L* at that hue
        Lc, Cc, a_r, b_r = chroma_envelope_vs_L_for_hue(
            Lab_ref, hue_deg, chroma, target_hue, tol_deg=tol_deg, L_min=0, L_max=100, L_step=1.0
        )
        fig = add_hue_ridge_to_fig(fig, Lc, a_r, b_r,
                                   scale=10.0,
                                   name=f'Max-C* ridge at h={target_hue:.1f}°',
                                   color='crimson')

    # C) Add RGB_{EBU} gamut surface
    fig = add_ebu_mesh(fig, n=41, color='rgba(0,120,255,0.30)', name_prefix='RGB_EBU')

    # Show
    try:
        fig.show()
    except Exception as e:
        import webbrowser, tempfile, pathlib
        print(f"Plotly inline show failed: {e}\nSaving to HTML and opening in browser.")
        out_path = pathlib.Path(tempfile.gettempdir()) / 'rosch_macadam_with_mesh_and_ebu.html'
        fig.write_html(str(out_path), include_plotlyjs='cdn')
        webbrowser.open(out_path.as_uri())

# %%
# Example: query hue 30° with ±1° tolerance
target_hue = np.hypot(-86.18, 81.18)
tol_deg = 1.0

res = max_chroma_stats_at_hue(Lab_ref, hue_deg, chroma, target_hue, tol_deg=tol_deg)
if res is None:
    print(f"No surface points found near h={target_hue}° (±{tol_deg}°).")
else:
    print(
        f"h*={res['h']:.2f}°, C*={res['C']:.2f}, L*={res['L']:.2f}, "
        f"a*={res['a']:.2f}, b*={res['b']:.2f}, Y/Yn={res['Y_rel']:.5f}"
    )

# Optional: mark it in your Plotly figure (same scale=10 as your plot)
fig.add_trace(go.Scatter3d(
    x=[res['a']], y=[res['b']], z=[res['L']],
    mode='markers',
    marker=dict(size=7, color='crimson', symbol='x'),
    name=f"Max C* @ h={target_hue:.1f}° (C*={res['C']:.1f}, L*={res['L']:.1f})",
    hovertemplate=(
        "h*: %{customdata[0]:.2f}°<br>"
        "C*: %{customdata[1]:.2f}<br>"
        "L*: %{z:.2f}<br>"
        "Y/Yn: %{customdata[2]:.5f}<extra></extra>"
    ),
    customdata=np.array([[res['h'], res['C'], res['Y_rel']]])
))
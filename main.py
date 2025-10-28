# %%
import sys
sys.path.append('C:/Users/SchwarzN/OneDrive - Université de Fribourg/Institution/P1_BraggSpericalPigments/Simulation/Rosch_MacAdam_OptimalColours')
import core
import plotting
import mesh
import analysis
import cache
import importlib
# # Reload the module to apply changes
importlib.reload(core)
importlib.reload(plotting)
importlib.reload(mesh)
importlib.reload(analysis)
importlib.reload(cache)
from cache import save_colour_solid_to_csv, load_colour_solid_from_csv
from analysis import max_chroma_at_hue, chroma_envelope_vs_L_for_hue, Lstar_to_Yrel, max_chroma_per_hue
from mesh import macadam_mesh_in_Lab_from_XYZ_hull, add_mesh3d, add_ebu_mesh
from core import build_rosch_macadam_XYZ
from plotting import plot_macadam_plotly
import os
import numpy as np
import plotly.graph_objects as go
import pandas as pd

import tempfile
import pathlib
import webbrowser
import time
import imageio


CACHE_NAME = "rosch_macadam_colour_solid.csv"
SCRIPT_DIR = "C:/Users/SchwarzN/OneDrive - Université de Fribourg/Institution/P1_BraggSpericalPigments/Simulation/Rosch_MacAdam_OptimalColours"
CACHE_FILE = SCRIPT_DIR + '/' + CACHE_NAME

def main():
    if os.path.exists(CACHE_FILE):
        XYZ_rel, Lab_ref, rgb_bright, hue_deg, chroma = load_colour_solid_from_csv(CACHE_FILE)
        print(f"Loaded colour solid data from cache: {CACHE_FILE}")
    else:
        XYZ_rel, Lab_ref, _, rgb_bright, hue_deg, chroma, _ = build_rosch_macadam_XYZ(spectral_res=1) # if spectral_res=None, default is 5 nm, else only 1,5,10,20 nm allowed
        save_colour_solid_to_csv(CACHE_FILE, XYZ_rel, Lab_ref, rgb_bright, hue_deg, chroma)
        print(f"Computed and cached colour solid data to: {CACHE_FILE}")

    fig = plot_macadam_plotly(Lab_ref, rgb_bright, hue_deg=hue_deg, chroma=chroma, opacity=0.1)

    V_plot, F = macadam_mesh_in_Lab_from_XYZ_hull(XYZ_rel, Lab_ref, scale=1.0)
    fig = add_mesh3d(fig, V_plot, F, color='rgba(80,80,80,0.18)', name='Optimal Colours Mesh')

    # Load data from rosch_macadam_colour_solid_1nm.csv
    df_rosch = pd.read_csv(CACHE_FILE)
    # Find the maximum chroma and their corresponding lightness for each hue angle in df_rosch
    df_max_chroma = max_chroma_per_hue(df_rosch, hue_resolution=1.0)
    # Save df_max_chroma to CSV
    max_chroma_csv = SCRIPT_DIR + '/' + 'rosch_macadam_max_chroma_per_hue.csv'
    df_max_chroma.to_csv(max_chroma_csv, index=False)
    # Add max chroma points to the plot
    fig.add_trace(go.Scatter3d(
        x=df_max_chroma['a_smooth'],
        y=df_max_chroma['b_smooth'],
        z=df_max_chroma['L_smooth'],
        mode='markers',
        marker=dict(size=5,color=df_max_chroma['hex_smooth'], opacity=1, line=dict(color='black', width=1)),
        name='Max Chroma per Hue'
    ))

    target_hue = 244 # in degrees
    tol_deg = 0.75
    peak = None
    # Calculate max chroma, lightness and luminance at target hue
    # peak = max_chroma_at_hue(Lab_ref, hue_deg, chroma, target_hue, tol_deg=tol_deg)
    if peak is not None:
        peak['Y_rel'] = Lstar_to_Yrel(peak['L'])
        fig.add_trace(go.Scatter3d(
            x=[peak['a']], y=[peak['b']], z=[peak['L']],
            mode='markers',
            marker=dict(size=7, color='black', symbol='x'),
            name=f'Max C* at h={target_hue:.1f}° (C*={peak["C"]:.1f}, L*={peak["L"]:.1f})',
            hovertemplate='h*: %{customdata[0]:.2f}°<br>C*: %{customdata[1]:.2f}<br>L*: %{z:.2f}<extra></extra>',
            customdata=np.array([[peak['h'], peak['C']]])
        ))
        print(f"Max chroma at h={target_hue}°: C*={peak['C']:.2f}, L*={peak['L']:.2f}, Y={peak['Y_rel']:.4f}")

        Lc, Cc, a_r, b_r = chroma_envelope_vs_L_for_hue(
            Lab_ref, hue_deg, chroma, target_hue, tol_deg=tol_deg
        )
        from analysis import add_hue_ridge_to_fig
        fig = add_hue_ridge_to_fig(fig, Lc, a_r, b_r, scale=1.0,
                                   name=f'Max-C* ridge at h={target_hue:.1f}°',
                                   color='black')

    # fig = add_ebu_mesh(fig, n=41, color='rgba(0,120,255,0.30)', name_prefix='RGB_EBU')
    
    # Save a gif of rotating the plot
    frame_dir = pathlib.Path("frames")
    frame_dir.mkdir(exist_ok=True)
    
    frames = []
    n_frames = 60 # 6° per frame for a full 360° rotation
    for i in range(n_frames):
        angle = i * (360 / n_frames)
        fig.update_layout(scene_camera=dict(
            eye=dict(x=2.5 * np.cos(np.radians(angle)), y=2.5 * np.sin(np.radians(angle)), z=1.5)
        ))
        frame_path = frame_dir / f"frame_{i:03d}.png"
        fig.write_image(str(frame_path), width=800, height=800)
        frames.append(imageio.v2.imread(str(frame_path)))

    # Save as gif
    gif_path = SCRIPT_DIR + '/' + 'GIF/rosch_macadam_colour_solid_rotation.gif'
    fps = 12
    duration = int(n_frames / fps)
    imageio.mimsave(gif_path, frames, duration=duration, loop=0)

    try:
        fig.show()
    except Exception as e:
        print(f"Plotly inline show failed: {e}\nSaving to HTML and opening in browser.")
        out_path = pathlib.Path(tempfile.gettempdir()) / 'rosch_macadam_with_mesh_and_ebu.html'
        fig.write_html(str(out_path), include_plotlyjs='cdn')
        time.sleep(1)  # Ensure file is written before opening
        webbrowser.open(out_path.as_uri())

if __name__ == "__main__":
    main()

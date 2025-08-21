import sys
import os
# Add the src directory to sys.path if running as a script
if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    src_dir = script_dir
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from itertools import cycle
import numpy as np
from cap3d_matplotlib import parse_cap3d, create_block_vertices

def draw_components_plotly(mediums, conductors, z_slice=None, title_suffix="", show_mediums=True, show_conductors=True):
    fig = go.Figure()

    # Enhanced color schemes with better contrast
    
    medium_colors = cycle(['rgba(31,119,180,0.3)', 'rgba(44,160,44,0.3)', 'rgba(255,127,14,0.3)', 
                          'rgba(148,103,189,0.3)', 'rgba(140,86,75,0.3)', 'rgba(227,119,194,0.3)'])
    conductor_colors = cycle(['rgba(214,39,40,0.9)', 'rgba(227,119,194,0.9)', 'rgba(127,127,127,0.9)', 
                             'rgba(188,189,34,0.9)', 'rgba(23,190,207,0.9)', 'rgba(255,20,147,0.9)'])

    def add_box_to_fig(fig, verts, color, opacity, label, component_type):
        # Create wireframe edges
        faces = [
            [0, 1, 2, 3], [4, 5, 6, 7],  # top and bottom
            [0, 1, 5, 4], [2, 3, 7, 6],  # front and back
            [1, 2, 6, 5], [0, 3, 7, 4],  # left and right
        ]
        
        # Add wireframe
        for face in faces:
            x = [verts[i][0] for i in face] + [verts[face[0]][0]]
            y = [verts[i][1] for i in face] + [verts[face[0]][1]]
            z = [verts[i][2] for i in face] + [verts[face[0]][2]]
            fig.add_trace(go.Scatter3d(
                x=x, y=y, z=z,
                mode='lines',
                line=dict(color='rgba(0,0,0,0.4)', width=1),
                hoverinfo='skip',
                showlegend=False,
                visible=True if (component_type == 'medium' and show_mediums) or 
                               (component_type == 'conductor' and show_conductors) else False
            ))

        # Add solid mesh with better hover info
        x, y, z = zip(*verts)
        center = np.mean(verts, axis=0)
        size = np.ptp(verts, axis=0)
        
        hover_text = f"""
        <b>{label}</b><br>
        Type: {component_type.title()}<br>
        Center: ({center[0]:.2f}, {center[1]:.2f}, {center[2]:.2f})<br>
        Size: {size[0]:.2f} × {size[1]:.2f} × {size[2]:.2f}
        """
        
        fig.add_trace(go.Mesh3d(
            x=x, y=y, z=z,
            color=color,
            opacity=opacity,
            alphahull=0,
            hovertemplate=hover_text + "<extra></extra>",
            name=label,
            showscale=False,
            legendgroup=component_type,
            visible=True if (component_type == 'medium' and show_mediums) or 
                           (component_type == 'conductor' and show_conductors) else False
        ))

    # Process mediums
    if show_mediums:
        for m in mediums:
            verts = create_block_vertices(m['base'], m['v1'], m['v2'], m['hvec'])
            if z_slice is not None and all(v[2] >= z_slice for v in verts):
                continue
            add_box_to_fig(fig, verts, color=next(medium_colors), opacity=0.3, 
                          label=f"{m['name']} / {m['block_name']}", component_type='medium')

    # Process conductors
    if show_conductors:
        for c in conductors:
            verts = create_block_vertices(c['base'], c['v1'], c['v2'], c['hvec'])
            if z_slice is not None and all(v[2] >= z_slice for v in verts):
                continue
            add_box_to_fig(fig, verts, color=next(conductor_colors), opacity=0.8, 
                          label=f"{c['name']} / {c['block_name']}", component_type='conductor')

    # Enhanced layout with better controls
    fig.update_layout(
        scene=dict(
            xaxis_title='X Axis',
            yaxis_title='Y Axis',
            zaxis_title='Z Axis',
            bgcolor='rgba(240,240,240,0.1)',
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.5),
                center=dict(x=0, y=0, z=0)
            ),
            aspectmode='cube'
        ),
        title=dict(
            text=f'3D CAP3D Visualization{title_suffix}',
            x=0.5,
            font=dict(size=16)
        ),
        showlegend=True,
        legend=dict(
            itemsizing='constant',
            orientation='v',
            x=1.02,
            y=1,
            bgcolor='rgba(255,255,255,0.8)',
            bordercolor='rgba(0,0,0,0.2)',
            borderwidth=1
        ),
        margin=dict(l=0, r=150, t=50, b=0),
        width=1000,
        height=700
    )

    return fig

def create_interactive_dashboard(cap3d_file):
    """Create an interactive dashboard with controls"""
    # Parse the data
    window, mediums, conductors = parse_cap3d(cap3d_file)
    
    # Calculate z-range for slider
    all_z = []
    for m in mediums:
        verts = create_block_vertices(m['base'], m['v1'], m['v2'], m['hvec'])
        all_z.extend([v[2] for v in verts])
    for c in conductors:
        verts = create_block_vertices(c['base'], c['v1'], c['v2'], c['hvec'])
        all_z.extend([v[2] for v in verts])
    
    z_min, z_max = min(all_z), max(all_z)
    
    # Create figures for different views
    figs = {}
    
    # Full view
    figs['full'] = draw_components_plotly(mediums, conductors, z_slice=None, 
                                         title_suffix=" - Complete View")
    
    # Mediums only
    figs['mediums'] = draw_components_plotly(mediums, conductors, z_slice=None, 
                                            title_suffix=" - Mediums Only", 
                                            show_mediums=True, show_conductors=False)
    
    # Conductors only  
    figs['conductors'] = draw_components_plotly(mediums, conductors, z_slice=None,
                                               title_suffix=" - Conductors Only",
                                               show_mediums=False, show_conductors=True)
    
    # Z-sliced view
    mid_z = (z_min + z_max) / 2
    figs['sliced'] = draw_components_plotly(mediums, conductors, z_slice=mid_z,
                                           title_suffix=f" - Z < {mid_z:.2f}")
    
    return figs, (z_min, z_max), len(mediums), len(conductors)

def show_visualization_menu(cap3d_file):
    """Show different visualization options"""
    figs, z_range, num_mediums, num_conductors = create_interactive_dashboard(cap3d_file)
    
    print(f"\n=== CAP3D Visualization Dashboard ===")
    print(f"File: {cap3d_file}")
    print(f"Components: {num_mediums} mediums, {num_conductors} conductors")
    print(f"Z-range: {z_range[0]:.2f} to {z_range[1]:.2f}")
    print("\nVisualization Options:")
    print("1. Complete view (all components)")
    print("2. Mediums only")
    print("3. Conductors only") 
    print("4. Z-sliced view")
    print("5. Show all views")
    
    return figs

# === Configuration ===
# Change this to your CAP3D file name
CAP3D_FILE = "0_120_38_30_89_MET1.cap3d"

# === Execution ===
if __name__ == "__main__":
    # Get the directory containing this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Look for the file in examples directory (one level up from src)
    cap3d_file = os.path.join(script_dir, "..", "examples", CAP3D_FILE)
    
    # Create dashboard
    figs = show_visualization_menu(cap3d_file)
    
    # Show complete view by default
    print("\nShowing complete view...")
    figs['full'].show()
    
    # Show conductors only for better visibility
    print("Showing conductors only...")
    figs['conductors'].show()
    
   
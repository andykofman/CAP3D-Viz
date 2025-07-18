import re
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np

# Utility for color cycling
from itertools import cycle


#--------- initialize parser ------------#

def parse_cap3d(file_path):

    """
    This function parses a CAP3D file into three main components:
        1. Window
        2. Medium
        3. Conductors


    """

    with open(file_path) as f:
        content = f.read()

    # parse window

    window_match = re.search(r"<window>.*?v1\((.*?)\).*?v2\((.*?)\)", content, re.DOTALL) # regex match
    if not window_match:
        raise ValueError("No window found in the file")

    """
    First, we need to parse the window.
    The window is a tuple of six numbers representing the coordinates of the window.
    The window is defined by two vertices, v1, v2 and hvector in the z-direction.

        
    <window>
        v1(1.0, 2.0, 3.0)
        v2(4.0, 5.0, 6.0)
    </window>
    
            
    After parsing:
    window_match.group(1)  is "1.0, 2.0, 3.0"
    window_match.group(2) is "4.0, 5.0, 6.0"

    """
    
    window = tuple (map(float, window_match.group(1).split(","))) + tuple(map(float, window_match.group(2).split(",")))


    """ 
    The above line does:
    Splits and converts "1.0, 2.0, 3.0" → (1.0, 2.0, 3.0)
    Splits and converts "4.0, 5.0, 6.0" → (4.0, 5.0, 6.0)
    Joins them: (1.0, 2.0, 3.0, 4.0, 5.0, 6.0) → window
    So, window will be a tuple containing all six numbers.

    """

    # parse medium

    mediums = []

    for m in re.finditer(r"<medium>(.*?)</medium>", content, re.DOTALL):

        # extract medium name and dielectric constant
        mtxt = m.group(1)
        name = re.search(r"name\s+(\S+)", mtxt)
        name = name.group(1) if name else "medium"
        diel = re.search(r"diel\s+([\d\.])", mtxt)
        diel = float(diel.group(1)) if diel else None
        
        for b in re.finditer(r"<block>(.*?)</block>", mtxt, re.DOTALL):

            # extract block name and dielectric constant
            btxt = b.group(1)
            block_name = re.search(r"name\s+(\S+)", btxt)
            block_name = block_name.group(1) if block_name else "block"
            base_match = re.search(r"basepoint\((.*?)\)", btxt)
            if not base_match:
                raise ValueError("No basepoint found in the block")
            base = tuple(map(float, base_match.group(1).split(",")))
            """
            get the vectors of the block (x, y, z)
            """
            v1_match = re.search(r"v1\((.*?)\)", btxt)
            v2_match = re.search(r"v2\((.*?)\)", btxt)
            hvec_match = re.search(r"hvector\((.*?)\)", btxt)
            if not (v1_match and v2_match and hvec_match):
                raise ValueError("No vectors found in the block")
            v1 = tuple(map(float, v1_match.group(1).split(",")))
            v2 = tuple(map(float, v2_match.group(1).split(",")))
            hvec = tuple(map(float, hvec_match.group(1).split(",")))

            """
            create a dictionary for the mediums
            """
            
            mediums.append({
                'type': 'medium',
                'name': name,
                'block_name': block_name,
                'base': base,
                'v1': v1,
                'v2': v2,
                'hvec': hvec,
                'diel': diel
            })

    # parse conductors

    conductors = []
    for m in re.finditer(r"<conductor>(.*?)</conductor>", content, re.DOTALL):
        mtxt = m.group(1)
        name = re.search(r"name\s+(\S+)", mtxt)
        name = name.group(1) if name else "conductor"
        for b in re.finditer(r"<block>(.*?)</block>", mtxt, re.DOTALL):
            btxt = b.group(1)
            block_name = re.search(r"name\s+(\S+)", btxt)
            block_name = block_name.group(1) if block_name else "block"
            base_match = re.search(r"basepoint\((.*?)\)", btxt)
            if not base_match:
                continue  # or raise ValueError("basepoint not found in block")
            base = tuple(map(float, base_match.group(1).split(',')))
            v1_match = re.search(r"v1\((.*?)\)", btxt)
            v2_match = re.search(r"v2\((.*?)\)", btxt)
            hvec_match = re.search(r"hvector\((.*?)\)", btxt)
            if not (v1_match and v2_match and hvec_match):
                continue  # or raise ValueError("v1/v2/hvector not found in block")
            v1 = tuple(map(float, v1_match.group(1).split(',')))
            v2 = tuple(map(float, v2_match.group(1).split(',')))
            hvec = tuple(map(float, hvec_match.group(1).split(',')))
            conductors.append({
                'type': 'conductor',
                'name': name,
                'block_name': block_name,
                'base': base,
                'v1': v1,
                'v2': v2,
                'hvec': hvec,
                'diel': None
            })

    return window, mediums, conductors


# ----------- Geometry ------------#
def create_block_vertices(base, v1, v2, hvec):
    """
    Create the vertices of a block. See block_animation.py for more details.
    """

    x,y,z = base
    dx, _, _ = v1
    _, dy, _ = v2
    _, _, dz = hvec

    # 8 vertices of a box

    return [
        (x, y, z), (x+dx, y, z), (x+dx, y+dy, z), (x, y+dy, z),
        (x, y, z+dz), (x+dx, y, z+dz), (x+dx, y+dy, z+dz), (x, y+dy, z+dz)
    ]

# ------------ Plotting ------------#

def draw_components(mediums, conductors, z_slice=None, title_suffix=""):
    """

    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # color cycles for mediums and conductors
    medium_colors = cycle(['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'])

    conductor_colors = cycle(['#d62728', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf', '#00bfff', '#ff1493', '#228b22', '#ffa500', '#8b0000'])

    # collet all boxes for sorting and axis limits
    all_boxes = []
    for m in mediums:
        verts = create_block_vertices(m['base'], m['v1'], m['v2'], m['hvec'])
        color = next(medium_colors)
        all_boxes.append({
            'type': 'medium',
            'verts': verts,
            'color': color,
            'label': f"{m['name']}\n{m['block_name']}",
            'alpha': 0.12  # very transparent for mediums
        })
    
    for c in conductors:
        verts = create_block_vertices(c['base'], c['v1'], c['v2'], c['hvec'])
        color = next(conductor_colors)
        all_boxes.append({
            'type': 'conductor',
            'verts': verts,
            'color': color,
            'label': f"{c['name']}\n{c['block_name']}",
            'alpha': 0.85  # more opaque for conductors
        })


    # Optionally filter by z_slice
    if z_slice is not None:
        def box_below_z(box):
            verts = np.array(box['verts'])
            return np.all(verts[:,2] < z_slice)
        all_boxes = [box for box in all_boxes if box_below_z(box)]
        if not all_boxes:
            print(f"No objects found below z={z_slice}")
            return
            
    # Compute axis limits
    all_verts = np.array([v for box in all_boxes for v in box['verts']])
    min_xyz = all_verts.min(axis=0)
    max_xyz = all_verts.max(axis=0)
    margin = 0.1 * (max_xyz - min_xyz)
    ax.set_xlim(min_xyz[0] - margin[0], max_xyz[0] + margin[0])
    ax.set_ylim(min_xyz[1] - margin[1], max_xyz[1] + margin[1])
    ax.set_zlim(min_xyz[2] - margin[2], max_xyz[2] + margin[2]) #type: ignore

    # Camera/view settings
    ax.view_init(elev=20, azim=30) #type: ignore

    # Sort boxes by distance from camera (back to front)
    def box_center(box):
        return np.mean(np.array(box['verts']), axis=0)
    def cam_dist(box):
        """
            This function calculates how "far" the center of the
            box is from the camera,
            in the direction the camera is looking.
        """
        elev, azim = np.deg2rad(20), np.deg2rad(30) # elevation and azimuth
        """
        cam_vec is a vector pointing in the direction
         the camera is looking, based on the above angles.
        """
        cam_vec = np.array([
            np.cos(elev)*np.cos(azim), 
            np.cos(elev)*np.sin(azim), 
            np.sin(elev)
        ])
        return np.dot(box_center(box), cam_vec)
    all_boxes.sort(key=cam_dist, reverse=True)  # back to front


    # Draw mediums first then conductors
    for typ in ['medium', 'conductor']:
        for box in [b for b in all_boxes if b['type'] == typ]:
            verts = box['verts']
            # Get vertices and define faces (6 faces for a box) defined by the corner

            faces = [
                [verts[i] for i in [0,1,2,3]],
                [verts[i] for i in [4,5,6,7]],
                [verts[i] for i in [0,1,5,4]],
                [verts[i] for i in [2,3,7,6]],
                [verts[i] for i in [1,2,6,5]],
                [verts[i] for i in [0,3,7,4]],
            ]

            # Draw faces
            pc = Poly3DCollection(faces, alpha=box['alpha'], facecolor=box['color'], edgecolor='k', linewidths=1.2 if typ=="conductor" else 0.7)
            ax.add_collection3d(pc)  #type: ignore
            # Draw edges (wireframe) on top connecting the corners in order and looping back to the start.
            for face in faces:
                xs, ys, zs = zip(*face + [face[0]])
                ax.plot(xs, ys, zs, color='k', linewidth=2.0 if typ=="conductor" else 1.0, alpha=0.95 if typ=="conductor" else 0.7, zorder=20)

            # Add labels (conductor only)

            if typ == "conductor":
                verts_arr = np.array(verts)
                center = np.mean(verts_arr, axis=0)
                box_height = np.max(verts_arr[:,2]) - np.min(verts_arr[:,2])
                box_width = np.max(verts_arr[:,0]) - np.min(verts_arr[:,0])
                label_pos = center.copy()
                label_pos[2] += 0.5 * box_height if box_height > 0 else 0.5
                label_pos[0] += 0.5 * box_width if box_width > 0 else 0.5
                ax.text(label_pos[0], label_pos[1], label_pos[2], box['label'], color='k', fontsize=8, ha='left', va='bottom', zorder=30)
                ax.plot([center[0], label_pos[0]], [center[1], label_pos[1]], [center[2], label_pos[2]], color='k', linewidth=0.8, linestyle='--', alpha=0.7, zorder=25)
    # Legend (custom)
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#1f77b4', edgecolor='k', label='Medium', alpha=0.12),
        Patch(facecolor='#d62728', edgecolor='k', label='Conductor', alpha=0.85)
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=10)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z') #type: ignore
    ax.set_title('3D Visualization of cap3d Components' + title_suffix)
    plt.tight_layout()
    plt.show()


# === Run ===
cap3d_file = 'smallcaseD.cap3d'
window, mediums, conductors = parse_cap3d(cap3d_file)
# Full plot
draw_components(mediums, conductors, z_slice=None, title_suffix=" (All)")
# Sliced plot (z < 2.0)
draw_components(mediums, conductors, z_slice=2.0, title_suffix=" (z < 2.0)")


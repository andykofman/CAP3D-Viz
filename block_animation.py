"""
This script creates an animation of the construction of a block (only blocks).

The block is constructed step by step, with each step being a different color.

The block is drawn by basepoint (initial starting point), with two vectors (v1 and v2) and an hvector (z-direction)
The animation can be saved as a GIF file.
"""



import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
import matplotlib.animation as animation

# Data for the first 4 blocks from smallcaseD.cap3d
blocks_data = [
    {
        'name': 'med1_1',
        'basepoint': (-2, -2, 0),
        'v1': (4, 0, 0),
        'v2': (0, 4, 0),
        'hvector': (0, 0, 0.3),
        'color': 'lightblue'
    },
    {
        'name': 'med1_2', 
        'basepoint': (-2, -2, 0.3),
        'v1': (4, 0, 0),
        'v2': (0, 4, 0),
        'hvector': (0, 0, 0.04),
        'color': 'lightgreen'
    },
    {
        'name': 'med2_1',
        'basepoint': (-2, -2, 0.34),
        'v1': (4, 0, 0),
        'v2': (0, 4, 0),
        'hvector': (0, 0, 0.12),
        'color': 'lightyellow'
    },
    {
        'name': 'med2_2',
        'basepoint': (-2, -2, 0.46),
        'v1': (4, 0, 0),
        'v2': (0, 4, 0),
        'hvector': (0, 0, 0.10),
        'color': 'lightcoral'
    }
]

def create_block_vertices(basepoint, v1, v2, hvector):
    """Create all 8 vertices of a block from basepoint and vectors"""
    bp = np.array(basepoint)
    v1 = np.array(v1)
    v2 = np.array(v2)
    hv = np.array(hvector)
    
    # 8 vertices of the block
    vertices = [
        bp,              # basepoint
        bp + v1,         # basepoint + v1
        bp + v2,         # basepoint + v2
        bp + v1 + v2,    # basepoint + v1 + v2
        bp + hv,         # basepoint + hvector
        bp + v1 + hv,    # basepoint + v1 + hvector
        bp + v2 + hv,    # basepoint + v2 + hvector
        bp + v1 + v2 + hv # basepoint + v1 + v2 + hvector
    ]
    
    return vertices

def create_block_faces(vertices):
    """Create the 6 faces of the block from vertices"""
    faces = [
        [vertices[0], vertices[1], vertices[3], vertices[2]],  # bottom
        [vertices[4], vertices[5], vertices[7], vertices[6]],  # top
        [vertices[0], vertices[1], vertices[5], vertices[4]],  # front
        [vertices[2], vertices[3], vertices[7], vertices[6]],  # back
        [vertices[0], vertices[2], vertices[6], vertices[4]],  # left
        [vertices[1], vertices[3], vertices[7], vertices[5]]   # right
    ]
    return faces

# Set up the figure and 3D axis
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

# Animation parameters
total_frames = len(blocks_data) * 30  # 30 frames per block
frames_per_block = 30
frames_per_step = 5  # 5 frames for each step (basepoint, v1, v2, hvector, complete block, pause)

# Storage for completed blocks
completed_blocks = []

def animate(frame):
    ax.clear()
    artists = []  # List to store artists for return
    
    # Calculate which block we're currently building
    current_block_idx = frame // frames_per_block
    frame_in_block = frame % frames_per_block
    
    # Draw all previously completed blocks
    for i, (block_data, block_faces) in enumerate(completed_blocks[:current_block_idx]):
        poly = Poly3DCollection(block_faces, alpha=0.7, facecolor=block_data['color'], 
                               edgecolor='black', linewidth=0.5)
        ax.add_collection(poly)
        artists.append(poly)
    
    # If we're still within the block count
    if current_block_idx < len(blocks_data):
        block = blocks_data[current_block_idx]
        bp = np.array(block['basepoint'])
        v1 = np.array(block['v1'])
        v2 = np.array(block['v2'])
        hv = np.array(block['hvector'])
        
        # Determine which step we're in for this block
        step = frame_in_block // frames_per_step
        
        # Step 0: Show basepoint (red dot)
        if step >= 0:
            scatter = ax.scatter(*bp, color='red', s=100, label='Basepoint')
            artists.append(scatter)
            text1 = ax.text(bp[0], bp[1], bp[2] + 0.1, f'Basepoint\n{block["basepoint"]}', 
                           fontsize=8, ha='center') # False Positive Error
            artists.append(text1)
        
        # Step 1: Show v1 vector (blue arrow)
        if step >= 1:
            quiver1 = ax.quiver(bp[0], bp[1], bp[2], v1[0], v1[1], v1[2], 
                               color='blue', arrow_length_ratio=0.1, linewidth=3, label='v1 vector')
            artists.append(quiver1)
            end_v1 = bp + v1
            text2 = ax.text(end_v1[0], end_v1[1], end_v1[2] + 0.1, f'v1: {block["v1"]}', 
                           fontsize=8, ha='center', color='blue') # False Positive Error
            artists.append(text2)
        
        # Step 2: Show v2 vector (green arrow)
        if step >= 2:
            quiver2 = ax.quiver(bp[0], bp[1], bp[2], v2[0], v2[1], v2[2], 
                               color='green', arrow_length_ratio=0.1, linewidth=3, label='v2 vector')
            artists.append(quiver2)
            end_v2 = bp + v2
            text3 = ax.text(end_v2[0], end_v2[1], end_v2[2] + 0.1, f'v2: {block["v2"]}', 
                           fontsize=8, ha='center', color='green') # False Positive Error
            artists.append(text3)
        
        # Step 3: Show hvector (orange arrow)
        if step >= 3:
            quiver3 = ax.quiver(bp[0], bp[1], bp[2], hv[0], hv[1], hv[2], 
                               color='orange', arrow_length_ratio=0.1, linewidth=3, label='hvector')
            artists.append(quiver3)
            end_hv = bp + hv
            text4 = ax.text(end_hv[0], end_hv[1], end_hv[2] + 0.1, f'hvector: {block["hvector"]}', 
                           fontsize=8, ha='center', color='orange')# False Positive Error
            artists.append(text4)
        
        # Step 4: Show complete block
        if step >= 4:
            vertices = create_block_vertices(block['basepoint'], block['v1'], 
                                           block['v2'], block['hvector'])
            faces = create_block_faces(vertices)
            
            poly = Poly3DCollection(faces, alpha=0.7, facecolor=block['color'], 
                                   edgecolor='black', linewidth=1)
            ax.add_collection(poly)
            artists.append(poly)
            
            # Add this block to completed blocks when animation moves to next block
            if frame_in_block == frames_per_block - 1:
                completed_blocks.append((block, faces))
    
    # Set labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z') # type: ignore
    
    if current_block_idx < len(blocks_data):
        ax.set_title(f'Building Block: {blocks_data[current_block_idx]["name"]}\n'
                    f'Block {current_block_idx + 1} of {len(blocks_data)}', fontsize=14)
    else:
        ax.set_title('All Blocks Complete!', fontsize=14)
    
    # Set axis limits
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.set_zlim(0, 1)  # type: ignore
    
    # Set viewing angle
    ax.view_init(elev=20, azim=45) # type: ignore
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    return artists

# Create animation
anim = animation.FuncAnimation(fig, animate, frames=total_frames, interval=200, repeat=True)

# Show the animation
plt.tight_layout()
plt.show()

# Save as GIF (optional - uncomment if you want to save)
# anim.save('block_construction.gif', writer='pillow', fps=5) 
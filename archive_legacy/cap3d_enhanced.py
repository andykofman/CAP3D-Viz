"""
CAP3D Enhanced Parser - Optimized for Large Files

This module provides enhanced parsing and visualization capabilities
for large CAP3D files that may be too memory-intensive for the standard parser.

"""

import re
import select
from tracemalloc import start 
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from typing import Generator, Dict, List, Tuple, Optional
import time
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor # parallel threads
import json 

class Block:

    def __init__(self, name: str, type: str, parent_name: str, 
                 base, v1, v2, hvec, diel:Optional[float] = None):
        
        self.name = name 
        self.type = type # medium or conductor 
        self.parent_name = parent_name 
        self.diel = diel 

        # Ensure all vectors are numpy arrays

        self.base = np.array(base, dtype=np.float32)
        self.v1   = np.array(v1, dtype=np.float32)
        self.v2   = np.array(v2, dtype=np.float32)
        self.hvec = np.array(hvec, dtype=np.float32)

    @property 
    def vertices(self) -> np.ndarray:
        """Generate 8 vertices of the box efficently"""

        x,y,z = self.base
        dx, _, _ = self.v1
        _, dy, _ = self.v2
        _, _, dz = self.hvec

        # Vectorized vertex generation

        vertices = np.array ([
            [x,y,z], [x+dx,y,z], [x+dx,y+dy,z], [x,y+dy,z],
            [x, y, z+dz], [x+dx, y, z+dz], [x+dx, y+dy, z+dz], [x, y+dy, z+dz]
        ], dtype=np.float32)

        return vertices
        

    @property 
    def bounds(self) -> Tuple[np.ndarray, np.ndarray]: 
        """ Get bounding box (min, max)"""
        vertices = self.vertices
        return vertices.min(axis=0), vertices.max(axis=0)

    @property
    def center(self) -> np.ndarray:
        """Get center point for LOD calculations"""
        return self.base + 0.5 * (self.v1 + self.v2 + self.hvec)
    
    @property
    def volume(self) -> float:
        """Calculate volume for LOD prioritization"""
        return float(abs(np.dot(self.v1, np.cross(self.v2, self.hvec))))
        



class StreamingCap3DParser:
    """ Memory-efficient streaming parser for large cap3d files """
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.stats = {
            'total_blocks': 0,
            'conductors': 0,
            'mediums': 0,
            'parse_time': 0
        }

    def parse_blocks_streaming(self) -> Generator[Block, None, None]:
        """True streaming parser - processes file line by line"""
        start_time = time.time()
        
        with open(self.file_path, 'r', encoding='utf-8', buffering=8192) as f:
            current_section = None  # 'medium' or 'conductor'
            current_section_name = None
            current_diel = None
            in_block = False
            block_data = {}
            
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line or line.startswith('<!--'):
                    continue
                
                # Section start
                if line.startswith('<medium>'):
                    current_section = 'medium'
                    current_section_name = None
                    current_diel = None
                elif line.startswith('<conductor>'):
                    current_section = 'conductor'
                    current_section_name = None
                    current_diel = None
                elif line.startswith('</medium>') or line.startswith('</conductor>'):
                    current_section = None
                    current_section_name = None
                    current_diel = None
                
                # Section properties
                elif current_section and line.startswith('name '):
                    current_section_name = line[5:].strip()
                elif current_section == 'medium' and line.startswith('diel '):
                    current_diel = float(line[5:].strip())
                
                # Block handling
                elif line.startswith('<block>'):
                    in_block = True
                    block_data = {'section_type': current_section, 'section_name': current_section_name, 'diel': current_diel}
                elif line.startswith('</block>') and in_block:
                    in_block = False
                    if self._is_valid_block(block_data):
                        block = self._create_block(block_data)
                        if block:
                            self.stats['total_blocks'] += 1
                            if block.type == 'medium':
                                self.stats['mediums'] += 1
                            else:
                                self.stats['conductors'] += 1
                            yield block
                    block_data = {}
                
                # Block properties
                elif in_block:
                    if line.startswith('name '):
                        block_data['name'] = line[5:].strip()
                    elif line.startswith('basepoint(') and line.endswith(')'):
                        coords_str = line[10:-1]  # Remove 'basepoint(' and ')'
                        block_data['base'] = self._parse_coords(coords_str)
                    elif line.startswith('v1(') and line.endswith(')'):
                        coords_str = line[3:-1]  # Remove 'v1(' and ')'
                        block_data['v1'] = self._parse_coords(coords_str)
                    elif line.startswith('v2(') and line.endswith(')'):
                        coords_str = line[3:-1]  # Remove 'v2(' and ')'
                        block_data['v2'] = self._parse_coords(coords_str)
                    elif line.startswith('hvector(') and line.endswith(')'):
                        coords_str = line[8:-1]  # Remove 'hvector(' and ')'
                        block_data['hvec'] = self._parse_coords(coords_str)
        
        self.stats['parse_time'] = time.time() - start_time

    def _parse_coords(self, coords_str: str) -> List[float]:
        """Fast coordinate parsing without regex"""
        try:
            return [float(x.strip()) for x in coords_str.split(',')]
        except ValueError:
            return [0.0, 0.0, 0.0]

    def _is_valid_block(self, block_data: dict) -> bool:
        """Check if block has all required fields"""
        required = ['section_type', 'section_name', 'base', 'v1', 'v2', 'hvec']
        return all(key in block_data for key in required)

    def _create_block(self, block_data: dict) -> Optional[Block]:
        """Create Block object from parsed data"""
        try:
            return Block(
                name=block_data.get('name', f"block_{self.stats['total_blocks']}"),
                type=block_data['section_type'],
                parent_name=block_data['section_name'],
                base=block_data['base'],
                v1=block_data['v1'],
                v2=block_data['v2'],
                hvec=block_data['hvec'],
                diel=block_data.get('diel')
            )
        except (ValueError, KeyError) as e:
            print(f"Warning: failed to create block: {e}")
            return None

    # Keep old method name for backward compatibility
    def parse_blocks_straming(self) -> Generator[Block, None, None]:
        """Backward compatibility wrapper"""
        return self.parse_blocks_streaming()

            
# color schemes
from itertools import cycle

class OptimizedCap3DVisualizer:
    """ High-performance 3D visualizer with LOD and filtering """
    def __init__ (self, max_blocks_display: int = 20000):
        self.max_blocks_display = max_blocks_display
        self.blocks = []
        self.bounds = None

        # Enhanced color schemes with better contrast
        self.medium_colors = ['rgba(31,119,180,0.3)', 'rgba(44,160,44,0.3)', 'rgba(255,127,14,0.3)', 
                            'rgba(148,103,189,0.3)', 'rgba(140,86,75,0.3)', 'rgba(227,119,194,0.3)']
        self.conductor_colors = ['rgba(214,39,40,0.9)', 'rgba(227,119,194,0.9)', 'rgba(127,127,127,0.9)', 
                                'rgba(188,189,34,0.9)', 'rgba(23,190,207,0.9)', 'rgba(255,20,147,0.9)']

    def load_data(self, file_path:str, progress_callback=None):
        """  Load data with progress tracking """
        parser = StreamingCap3DParser(file_path)
        print(f"Loading {file_path}...")

        start_time = time.time()
        self.blocks = []

        for i, block in enumerate(parser.parse_blocks_streaming()):
            self.blocks.append(block)

            if progress_callback and i % 1000 == 0:
                progress_callback(i)
        
        load_time = time.time() - start_time
        print(f"Loaded {len(self.blocks)} blocks in {load_time:.2f}s")

        print(f"Parser stats: {parser.stats}")

        # calculate global bounds
        self._calculate_bounds()
        
    def _calculate_bounds(self):
        """ helper funct to calculate bounding box for all blocks """

        if not self.blocks:
            return
        all_mins = []
        all_maxs = []

        for block in self.blocks[:1000]: # Sample first 1000 blocks for bounds 
            min_bound, max_bound = block.bounds
            all_maxs.append(max_bound)
            all_mins.append(min_bound)

        all_mins = np.array(all_mins)
        all_maxs = np.array(all_maxs)

        self.bounds = (all_mins.min(axis=0), all_maxs.max(axis=0))
        print(f"Global bounds: {self.bounds}")

    
    def filter_blocks(self,
                     show_mediums: bool = True,
                     show_conductors: bool = True,
                     z_min: Optional[float] = None,         # If set, only include blocks whose minimum z-coordinate (vertical position) is greater than or equal to this value.
                     z_max: Optional[float] = None,         # If set, only include blocks whose maximum z-coordinate is less than or equal to this value.
                     spatial_filter: Optional[Tuple[np.ndarray, np.ndarray]] = None, # If set, should be a tuple (min_xyz, max_xyz) where each is a numpy array of shape (3,).
                     volume_threshold: Optional[float] = None # If set, only include blocks whose volume (as computed by the Block class) is greater than or equal to this value.
                    
                      ) ->List[Block]:

            """Advance filtering with multiple criteria"""

            filtered = []

            for block in self.blocks:
                #type filter
                if block.type == 'medium' and not show_mediums:
                    continue

                if block.type == 'conductor' and not show_conductors:
                    continue

                # Z-slice filter

                min_bound, max_bound = block.bounds
                if z_min is not None and max_bound[2] < z_min:
                    continue
                if z_max is not None and min_bound[2] > z_max:
                    continue
                

               # Spatial filter (bounding box)

                if spatial_filter is not None:
                    filter_min, filter_max = spatial_filter
                    if (max_bound < filter_min).any() or (min_bound > filter_max).any():
                        continue
                
                # Volume filter (for LOD)
                if volume_threshold is not None and block.volume < volume_threshold:
                    continue

                filtered.append(block)
            return filtered



    def create_optimized_visualization(self, 
                                       show_mediums: bool = True,
                                       show_conductors: bool = True,
                                       z_slice: Optional[float] =None,                                     
                                       max_blocks: Optional[int] = None,
                                       use_lod: bool = True,
                                       show_edges: bool = True,
                                       opacity_mediums: float = 0.3,
                                       opacity_conductors: float = 0.9) -> go.Figure:
        """Create optimized plotly visualization"""

        print("Filtering blocks....")
        filtered_blocks = self.filter_blocks(
            show_mediums = show_mediums,
            show_conductors = show_conductors,
            z_max = z_slice
        )

        # Apply LOD if needed

        if use_lod and max_blocks and len(filtered_blocks) > max_blocks:
            filtered_blocks = self._apply_lod(filtered_blocks, max_blocks)
        
        print(f"Visaualizing {len(filtered_blocks)} blocks...")

        fig = go.Figure()

        # Group blocks by type for efficient rendering

        mediums = [b for b in filtered_blocks if b.type == 'medium']
        conductors = [b for b in filtered_blocks if b.type == 'conductor']

        # Render mediums
        if mediums and show_mediums:
            self._add_blocks_to_figure(fig, mediums, opacity_mediums, 'medium', show_edges)

        # Render conductors
        if conductors and show_conductors:
            self._add_blocks_to_figure(fig, conductors, opacity_conductors, 'conductor', show_edges)
        

        # Configure layout
        fig.update_layout(
            scene=dict(
                xaxis_title='X (mm)',
                yaxis_title='Y (mm)', 
                zaxis_title='Z (mm)',
                aspectmode='data'
            ),
            title=f'Optimized 3D Cap3D Visualization ({len(filtered_blocks)} blocks)',
            showlegend=True,
            legend=dict(itemsizing='constant'),
            width=1200,
            height=800
        )
        
        return fig      


    def _apply_lod(self, blocks: List[Block], max_blocks: int) -> List[Block]:
        """ Apply Level of Detail - prioritize larger, more important blocks """
        if len(blocks) <= max_blocks:
            return blocks
        # Sort by vloume (Larger blocks are more important)

        blocks_with_priority = [(block, block.volume) for block in blocks]
        blocks_with_priority.sort (key=lambda x:x[1], reverse=True)

        # Take top blocks by volume 
        selected = [block for block, _ in blocks_with_priority[:max_blocks]]

                
        print(f"LOD: Reduced from {len(blocks)} to {len(selected)} blocks")
        return selected
    
    def _add_blocks_to_figure(self, fig: go.Figure, blocks: list, 
                            opacity: float, block_type: str, show_edges: bool):
        """Efficiently add blocks to figure using per-block color cycling for visual distinction"""
        if not blocks:
            return

        colors = self.medium_colors if block_type == 'medium' else self.conductor_colors
        num_colors = len(colors)
        for idx, block in enumerate(blocks):
            color = colors[idx % num_colors]
            vertices = block.vertices
            # Define faces using vertex indices
            faces = [
                [0, 1, 2], [0, 2, 3],  # bottom
                [4, 7, 6], [4, 6, 5],  # top
                [0, 4, 5], [0, 5, 1],  # front
                [2, 6, 7], [2, 7, 3],  # back
                [1, 5, 6], [1, 6, 2],  # right
                [0, 3, 7], [0, 7, 4],  # left
            ]
            i, j, k = [], [], []
            for face in faces:
                i.append(face[0])
                j.append(face[1])
                k.append(face[2])
            fig.add_trace(go.Mesh3d(
                x=vertices[:, 0],
                y=vertices[:, 1],
                z=vertices[:, 2],
                i=i,
                j=j,
                k=k,
                color=color,
                opacity=opacity,
                name=f'{block_type.title()} {block.name}',
                showscale=False,
                hovertemplate=f'<b>{block_type.title()} {block.name}</b><br>X: %{{x}}<br>Y: %{{y}}<br>Z: %{{z}}<extra></extra>'
            ))

    def create_interactive_dashboard(self):
        """Create an interactive dashboard with controls"""
        # This would create a Dash app with controls
        # For now, return a basic figure with buttons
        fig = self.create_optimized_visualization()
        
        # Add filter buttons
        fig.update_layout(
            updatemenus=[
                dict(
                    type="buttons",
                    direction="left",
                    buttons=list([
                        dict(
                            args=[{"visible": [True, True]}],
                            label="Show All",
                            method="restyle"
                        ),
                        dict(
                            args=[{"visible": [True, False]}],
                            label="Mediums Only", 
                            method="restyle"
                        ),
                        dict(
                            args=[{"visible": [False, True]}],
                            label="Conductors Only",
                            method="restyle"
                        )
                    ]),
                    pad={"r": 10, "t": 10},
                    showactive=True,
                    x=0.01,
                    xanchor="left",
                    y=1.02,
                    yanchor="top"
                ),
            ]
        )
        
        return fig
    
    def export_to_html(self, fig: go.Figure, filename: str = "cap3d_visualization.html"):
        """Export visualization to HTML file"""
        fig.write_html(filename, include_plotlyjs='cdn')
        print(f"Visualization saved to {filename}")


# Convenience functions
def load_and_visualize(file_path: str, 
                      max_blocks: int = 10000,
                      z_slice: Optional[float] = None,
                      export_html: bool = True) -> go.Figure:
    """One-shot function to load and visualize a cap3d file"""
    visualizer = OptimizedCap3DVisualizer(max_blocks_display=max_blocks)
    visualizer.load_data(file_path)
    
    fig = visualizer.create_optimized_visualization(
        z_slice=z_slice,
        max_blocks=max_blocks,
        use_lod=True
    )
    
    if export_html:
        visualizer.export_to_html(fig, f"{file_path.replace('.cap3d', '_visualization.html')}")
    
    return fig

def quick_preview(file_path: str, max_blocks: int = 1000) -> go.Figure:
    """Quick preview with aggressive LOD for very large files"""
    return load_and_visualize(file_path, max_blocks=max_blocks, export_html=False)



if __name__ == "__main__":
    import os
    # Change this to your CAP3D file name
    CAP3D_FILE = "0_120_38_30_89_MET1.cap3d"
    # Get the directory containing this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Look for the file in examples directory (one level up from src)
    cap3d_file = os.path.join(script_dir, "..", "examples", CAP3D_FILE)

    print("Testing with small file first...")
    fig_large = quick_preview(cap3d_file, max_blocks=2000)
    fig_large.show() 

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


@dataclass
class CachedMesh:
    """Cached mesh data for efficient rendering"""
    x: np.ndarray # X coordinates of vertices
    y: np.ndarray  # Y coordinates of vertices
    z: np.ndarray  # Z coordinates of vertices
    i: List[int]    # Face connectivity indices
    j: List[int]
    k: List[int]
    block_type: str # medium or conductor
    block_index: int # original block index
    center: np.ndarray # center point of the block
    volume: float # volume of the block
    bounds: Tuple[np.ndarray, np.ndarray] # bounding box of the block
        



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
    """ High-performance 3D visualizer with LOD and caching """
    def __init__ (self, max_blocks_display: int = 20000):
        self.max_blocks_display = max_blocks_display
        self.blocks = []
        self.bounds = None
        
        # Caching system for performance
        self._cached_meshes: List[CachedMesh] = []
        self._mesh_cache_valid = False
        self._figure_cache: Optional[go.Figure] = None
        
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
        
        # Invalidate caches
        self._mesh_cache_valid = False
        self._figure_cache = None
        
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

    def _build_mesh_cache(self, max_blocks: Optional[int] = None, use_lod: bool = True):
        """Build cached mesh data for all blocks once"""
        print("Building mesh cache...")
        start_time = time.time()
        
        # Apply LOD if needed
        blocks_to_cache = self.blocks
        if use_lod and max_blocks and len(blocks_to_cache) > max_blocks:
            blocks_to_cache = self._apply_lod(blocks_to_cache, max_blocks)
        
        self._cached_meshes = []
        for idx, block in enumerate(blocks_to_cache):
            if idx % 1000 == 0 and idx > 0:
                print(f"  Cached {idx}/{len(blocks_to_cache)} meshes...")
                
            vertices = block.vertices
            
            # Define faces using vertex indices (same as before)
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
            
            cached_mesh = CachedMesh(
                x=vertices[:, 0],
                y=vertices[:, 1], 
                z=vertices[:, 2],
                i=i,
                j=j,
                k=k,
                block_type=block.type,
                block_index=idx,
                center=block.center,
                volume=block.volume,
                bounds=block.bounds
            )
            self._cached_meshes.append(cached_mesh)
        
        self._mesh_cache_valid = True
        cache_time = time.time() - start_time
        print(f"Built mesh cache for {len(self._cached_meshes)} blocks in {cache_time:.2f}s")

    def _create_figure_with_all_traces(self) -> go.Figure:
        """Create figure with all traces, using visibility for filtering"""
        if not self._mesh_cache_valid:
            raise RuntimeError("Mesh cache not built. Call _build_mesh_cache() first.")
        
        print("Creating figure with all traces...")
        start_time = time.time()
        
        fig = go.Figure()
        
        # Group meshes by type and color
        medium_colors = cycle(self.medium_colors)
        conductor_colors = cycle(self.conductor_colors)
        
        # Add all traces at once
        for mesh in self._cached_meshes:
            if mesh.block_type == 'medium':
                color = next(medium_colors)
                opacity = 0.3
                name = f"Medium {mesh.block_index}"
            else:
                color = next(conductor_colors)
                opacity = 0.9
                name = f"Conductor {mesh.block_index}"
            
            fig.add_trace(go.Mesh3d(
                x=mesh.x,
                y=mesh.y,
                z=mesh.z,
                i=mesh.i,
                j=mesh.j,
                k=mesh.k,
                color=color,
                opacity=opacity,
                name=name,
                showscale=False,
                visible=True,  # All start visible
                hovertemplate=f'<b>{mesh.block_type.title()} {mesh.block_index}</b><br>X: %{{x}}<br>Y: %{{y}}<br>Z: %{{z}}<extra></extra>',
                # Store metadata for filtering
                customdata=[mesh.block_type, mesh.center[2], mesh.volume]
            ))
        
        # Configure layout once
        fig.update_layout(
            scene=dict(
                xaxis_title='X (mm)',
                yaxis_title='Y (mm)', 
                zaxis_title='Z (mm)',
                aspectmode='data'
            ),
            title=f'Cached 3D CAP3D Visualization ({len(self._cached_meshes)} blocks)',
            showlegend=False,  # Too many traces for useful legend
            width=1200,
            height=800
        )
        
        build_time = time.time() - start_time
        print(f"Created figure with {len(self._cached_meshes)} traces in {build_time:.2f}s")
        
        return fig
    
    def filter_blocks(self,
                     show_mediums: bool = True,
                     show_conductors: bool = True,
                     z_min: Optional[float] = None,         
                     z_max: Optional[float] = None,         
                     spatial_filter: Optional[Tuple[np.ndarray, np.ndarray]] = None,
                     volume_threshold: Optional[float] = None
                    
                      ) ->List[Block]:

            """Advanced filtering with multiple criteria"""

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

    def _create_filtered_figure_from_cache(self,
                                          show_mediums: bool = True,
                                          show_conductors: bool = True,
                                          z_min: Optional[float] = None,
                                          z_max: Optional[float] = None) -> go.Figure:
        """Create a new figure with only the filtered meshes from cache"""
        if not self._cached_meshes:
            raise RuntimeError("No cached meshes available. Build cache first.")
        
        print("Creating filtered figure from cache...")
        start_time = time.time()
        
        fig = go.Figure()
        
        # Filter and add only the relevant meshes
        medium_colors = cycle(self.medium_colors)
        conductor_colors = cycle(self.conductor_colors)
        visible_count = 0
        
        for mesh in self._cached_meshes:
            # Apply filters
            include = True
            
            # Type filter
            if mesh.block_type == 'medium' and not show_mediums:
                include = False
            elif mesh.block_type == 'conductor' and not show_conductors:
                include = False
            
            # Z-slice filter
            if include and z_min is not None:
                min_bound, max_bound = mesh.bounds
                if max_bound[2] < z_min or (z_max is not None and min_bound[2] > z_max):
                    include = False
            elif include and z_max is not None:
                min_bound, max_bound = mesh.bounds
                if min_bound[2] > z_max:
                    include = False
            
            if include:
                if mesh.block_type == 'medium':
                    color = next(medium_colors)
                    opacity = 0.3
                    name = f"Medium {mesh.block_index}"
                else:
                    color = next(conductor_colors)
                    opacity = 0.9
                    name = f"Conductor {mesh.block_index}"
                
                fig.add_trace(go.Mesh3d(
                    x=mesh.x,
                    y=mesh.y,
                    z=mesh.z,
                    i=mesh.i,
                    j=mesh.j,
                    k=mesh.k,
                    color=color,
                    opacity=opacity,
                    name=name,
                    showscale=False,
                    hovertemplate=f'<b>{mesh.block_type.title()} {mesh.block_index}</b><br>X: %{{x}}<br>Y: %{{y}}<br>Z: %{{z}}<extra></extra>'
                ))
                visible_count += 1
        
        # Configure layout
        fig.update_layout(
            scene=dict(
                xaxis_title='X (mm)',
                yaxis_title='Y (mm)', 
                zaxis_title='Z (mm)',
                aspectmode='data'
            ),
            title=f'Cached 3D CAP3D Visualization ({visible_count}/{len(self._cached_meshes)} blocks visible)',
            showlegend=False,  # Too many traces for useful legend
            width=1200,
            height=800
        )
        
        build_time = time.time() - start_time
        print(f"Created filtered figure with {visible_count} traces in {build_time:.3f}s")
        
        return fig

    def apply_filters_to_figure(self, fig: go.Figure,
                               show_mediums: bool = True,
                               show_conductors: bool = True,
                               z_min: Optional[float] = None,
                               z_max: Optional[float] = None) -> go.Figure:
        """Apply filters by toggling trace visibility instead of rebuilding"""
        
        if not self._cached_meshes:
            raise RuntimeError("No cached meshes available. Build cache first.")
        
        print("Applying filters via visibility toggle...")
        start_time = time.time()
        
        # Create visibility mask
        visibility_mask = []
        visible_count = 0
        
        for i, mesh in enumerate(self._cached_meshes):
            visible = True
            
            # Type filter
            if mesh.block_type == 'medium' and not show_mediums:
                visible = False
            elif mesh.block_type == 'conductor' and not show_conductors:
                visible = False
            
            # Z-slice filter
            if visible and z_min is not None:
                min_bound, max_bound = mesh.bounds
                if max_bound[2] < z_min or (z_max is not None and min_bound[2] > z_max):
                    visible = False
            elif visible and z_max is not None:
                min_bound, max_bound = mesh.bounds
                if min_bound[2] > z_max:
                    visible = False
            
            visibility_mask.append(visible)
            if visible:
                visible_count += 1
        
        # Update trace visibility individually
        for i, visible in enumerate(visibility_mask):
            fig.data[i].visible = visible
        
        # Update title
        fig.update_layout(title=f'Filtered 3D CAP3D Visualization ({visible_count}/{len(self._cached_meshes)} blocks visible)')
        
        filter_time = time.time() - start_time
        print(f"Applied filters to {visible_count}/{len(self._cached_meshes)} blocks in {filter_time:.3f}s")
        
        return fig

    def create_optimized_visualization(self, 
                                       show_mediums: bool = True,
                                       show_conductors: bool = True,
                                       z_slice: Optional[float] =None,                                     
                                       max_blocks: Optional[int] = None,
                                       use_lod: bool = True,
                                       show_edges: bool = True,
                                       opacity_mediums: float = 0.3,
                                       opacity_conductors: float = 0.9,
                                       use_cache: bool = True,
                                       use_batched_rendering: bool = True) -> go.Figure:
        """Create optimized plotly visualization using caching"""

        # Build cache if needed
        if use_cache and not self._mesh_cache_valid:
            self._build_mesh_cache(max_blocks=max_blocks, use_lod=use_lod)
        
        # Use new batched rendering for better performance
        if use_batched_rendering:
            return self._create_batched_visualization(
                show_mediums=show_mediums,
                show_conductors=show_conductors,
                z_slice=z_slice,
                opacity_mediums=opacity_mediums,
                opacity_conductors=opacity_conductors
            )
        elif use_cache:
            # Create new figure with cached meshes and apply filters
            fig = self._create_filtered_figure_from_cache(
                show_mediums=show_mediums,
                show_conductors=show_conductors,
                z_max=z_slice
            )
        else:
            # Fallback to old method (for comparison)
            print("Using non-cached visualization (slower)...")
            fig = self._create_visualization_legacy(
                show_mediums, show_conductors, z_slice, max_blocks, use_lod, show_edges, 
                opacity_mediums, opacity_conductors
            )
        
        return fig

    def _create_batched_visualization(self,
                                     show_mediums: bool = True,
                                     show_conductors: bool = True,
                                     z_slice: Optional[float] = None,
                                     opacity_mediums: float = 0.3,
                                     opacity_conductors: float = 0.9) -> go.Figure:
        """Create visualization using batched geometry - MUCH faster for large datasets"""
        
        if not self._cached_meshes:
            raise RuntimeError("No cached meshes available. Build cache first.")
        
        print("Creating batched visualization (high performance mode)...")
        start_time = time.time()
        
        fig = go.Figure()
        
        # Separate meshes by type for batching
        medium_meshes = []
        conductor_meshes = []
        
        for mesh in self._cached_meshes:
            # Apply z-slice filter
            if z_slice is not None:
                min_bound, max_bound = mesh.bounds
                if min_bound[2] > z_slice:
                    continue
            
            if mesh.block_type == 'medium' and show_mediums:
                medium_meshes.append(mesh)
            elif mesh.block_type == 'conductor' and show_conductors:
                conductor_meshes.append(mesh)
        
        # Create multiple batched traces with color cycling for variety
        if medium_meshes:
            self._add_color_batched_meshes(fig, medium_meshes, 'medium', opacity_mediums)
        
        if conductor_meshes:
            self._add_color_batched_meshes(fig, conductor_meshes, 'conductor', opacity_conductors)
        
        # Configure layout
        fig.update_layout(
            scene=dict(
                xaxis_title='X (mm)',
                yaxis_title='Y (mm)', 
                zaxis_title='Z (mm)',
                aspectmode='data'
            ),
            title=f'Color-Batched 3D CAP3D Visualization ({len(medium_meshes) + len(conductor_meshes)} blocks, {len(fig.data)} traces)',
            showlegend=True,
            width=1200,
            height=800
        )
        
        build_time = time.time() - start_time
        print(f"Created color-batched figure with {len(medium_meshes) + len(conductor_meshes)} blocks ({len(fig.data)} traces) in {build_time:.3f}s")
        
        return fig

    def _add_batched_meshes(self, fig: go.Figure, meshes: List[CachedMesh], 
                           block_type: str, opacity: float):
        """Combine multiple meshes into a single trace for performance"""
        
        if not meshes:
            return
        
        # Combine all vertices and faces
        all_x, all_y, all_z = [], [], []
        all_i, all_j, all_k = [], [], []
        vertex_offset = 0
        
        for mesh in meshes:
            # Add vertices
            all_x.extend(mesh.x)
            all_y.extend(mesh.y) 
            all_z.extend(mesh.z)
            
            # Add faces with proper vertex offset
            all_i.extend([idx + vertex_offset for idx in mesh.i])
            all_j.extend([idx + vertex_offset for idx in mesh.j])
            all_k.extend([idx + vertex_offset for idx in mesh.k])
            
            # Update offset for next mesh
            vertex_offset += len(mesh.x)
        
        # Choose color based on type using cycling pattern like other methods
        if block_type == 'medium':
            medium_colors = cycle(self.medium_colors)
            color = next(medium_colors).replace('0.3', '0.4')  # Slightly more opaque for batched
            name = f'Mediums ({len(meshes)} blocks)'
        else:
            conductor_colors = cycle(self.conductor_colors)
            color = next(conductor_colors)  # Keep original opacity
            name = f'Conductors ({len(meshes)} blocks)'
        
        # Create single batched trace
        fig.add_trace(go.Mesh3d(
            x=all_x,
            y=all_y,
            z=all_z,
            i=all_i,
            j=all_j,
            k=all_k,
            color=color,
            opacity=opacity,
            name=name,
            showscale=False,
            hovertemplate=f'<b>{block_type.title()}</b><br>X: %{{x}}<br>Y: %{{y}}<br>Z: %{{z}}<extra></extra>'
        ))

    def _add_color_batched_meshes(self, fig: go.Figure, meshes: List[CachedMesh], 
                                 block_type: str, opacity: float):
        """Create multiple batched traces with color cycling for visual variety"""
        
        if not meshes:
            return
            
        # Set up color cycling
        if block_type == 'medium':
            colors = cycle(self.medium_colors)
        else:
            colors = cycle(self.conductor_colors)
        
        # Create batches for color variety (balance performance vs visual variety)
        batch_size = max(1, len(meshes) // 6)  # Create ~6 color groups max
        
        batch_count = 0
        for i in range(0, len(meshes), batch_size):
            batch_meshes = meshes[i:i + batch_size]
            if not batch_meshes:
                continue
                
            # Get next color for this batch
            color = next(colors)
            if block_type == 'medium':
                color = color.replace('0.3', '0.4')  # Slightly more opaque for batched
            
            # Combine meshes in this batch
            all_x, all_y, all_z = [], [], []
            all_i, all_j, all_k = [], [], []
            vertex_offset = 0
            
            for mesh in batch_meshes:
                # Add vertices
                all_x.extend(mesh.x)
                all_y.extend(mesh.y) 
                all_z.extend(mesh.z)
                
                # Add faces with proper vertex offset
                all_i.extend([idx + vertex_offset for idx in mesh.i])
                all_j.extend([idx + vertex_offset for idx in mesh.j])
                all_k.extend([idx + vertex_offset for idx in mesh.k])
                
                # Update offset for next mesh
                vertex_offset += len(mesh.x)
            
            # Create batched trace for this color group
            batch_count += 1
            name = f'{block_type.title()}s-{batch_count} ({len(batch_meshes)} blocks)'
            
            fig.add_trace(go.Mesh3d(
                x=all_x,
                y=all_y,
                z=all_z,
                i=all_i,
                j=all_j,
                k=all_k,
                color=color,
                opacity=opacity,
                name=name,
                showscale=False,
                hovertemplate=f'<b>{block_type.title()}</b><br>X: %{{x}}<br>Y: %{{y}}<br>Z: %{{z}}<extra></extra>'
            ))

    def _create_visualization_legacy(self, show_mediums, show_conductors, z_slice, max_blocks, use_lod, show_edges, opacity_mediums, opacity_conductors):
        """Legacy visualization method for comparison"""
        print("Filtering blocks....")
        filtered_blocks = self.filter_blocks(
            show_mediums = show_mediums,
            show_conductors = show_conductors,
            z_max = z_slice
        )

        # Apply LOD if needed
        if use_lod and max_blocks and len(filtered_blocks) > max_blocks:
            filtered_blocks = self._apply_lod(filtered_blocks, max_blocks)
        
        print(f"Visualizing {len(filtered_blocks)} blocks...")

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
            title=f'Legacy 3D Cap3D Visualization ({len(filtered_blocks)} blocks)',
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

    def create_interactive_dashboard(self, use_batched: bool = True):
        """Create an interactive dashboard with controls"""
        
        # Build cache first for interactivity
        if not self._mesh_cache_valid:
            self._build_mesh_cache()
        
        if use_batched:
            # Create batched visualization (2 traces max)
            fig = self._create_batched_visualization()
            
            # Enhanced interactivity for batched traces
            fig.update_layout(
                updatemenus=[
                    dict(
                        type="buttons",
                        direction="left",
                        buttons=list([
                            dict(
                                args=[{"visible": [True] * len(fig.data)}],
                                label="Show All",
                                method="restyle"
                            ),
                            dict(
                                args=[{"visible": [True if 'Medium' in trace.name else False for trace in fig.data]}],
                                label="Mediums Only", 
                                method="restyle"
                            ),
                            dict(
                                args=[{"visible": [True if 'Conductor' in trace.name else False for trace in fig.data]}],
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
                    # Add opacity control
                    dict(
                        type="dropdown",
                        direction="down",
                        buttons=list([
                            dict(
                                args=[{"opacity": [0.1 if 'Medium' in trace.name else 0.9 for trace in fig.data]}],
                                label="Low Medium Opacity",
                                method="restyle"
                            ),
                            dict(
                                args=[{"opacity": [0.3 if 'Medium' in trace.name else 0.9 for trace in fig.data]}],
                                label="Medium Opacity",
                                method="restyle"
                            ),
                            dict(
                                args=[{"opacity": [0.6 if 'Medium' in trace.name else 0.9 for trace in fig.data]}],
                                label="High Medium Opacity",
                                method="restyle"
                            )
                        ]),
                        pad={"r": 10, "t": 10},
                        showactive=True,
                        x=0.15,
                        xanchor="left",
                        y=1.02,
                        yanchor="top"
                    ),
                ]
            )
        else:
            # Fallback to old method (will be slow with many blocks)
            fig = self.create_optimized_visualization(use_batched_rendering=False)
            
            # Basic interactivity for many-trace version (limited by performance)
            fig.update_layout(
                updatemenus=[
                    dict(
                        type="buttons",
                        direction="left",
                        buttons=list([
                            dict(
                                args=[{"visible": [True] * len(fig.data)}],
                                label="Show All",
                                method="restyle"
                            ),
                            dict(
                                args=[{"visible": [i % 2 == 0 for i in range(len(fig.data))]}],
                                label="Mediums Only", 
                                method="restyle"
                            ),
                            dict(
                                args=[{"visible": [i % 2 == 1 for i in range(len(fig.data))]}],
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

    def clear_cache(self):
        """Clear all caches to free memory"""
        self._cached_meshes = []
        self._mesh_cache_valid = False
        self._figure_cache = None
        print("Cleared all visualization caches")


# Convenience functions
def load_and_visualize(file_path: str, 
                      max_blocks: int = 100000,  # Increased default with batched rendering
                      z_slice: Optional[float] = None,
                      export_html: bool = True,
                      use_cache: bool = True,
                      use_batched: bool = True) -> go.Figure:
    """One-shot function to load and visualize a cap3d file"""
    visualizer = OptimizedCap3DVisualizer(max_blocks_display=max_blocks)
    visualizer.load_data(file_path)
    
    fig = visualizer.create_optimized_visualization(
        z_slice=z_slice,
        max_blocks=max_blocks,
        use_lod=True,
        use_cache=use_cache,
        use_batched_rendering=use_batched
    )
    
    if export_html:
        visualizer.export_to_html(fig, f"{file_path.replace('.cap3d', '_visualization.html')}")
    
    return fig

def quick_preview(file_path: str, max_blocks: int = 50000) -> go.Figure:
    """Quick preview - now handles much larger files with batched rendering"""
    return load_and_visualize(file_path, max_blocks=max_blocks, export_html=False)

def create_interactive_dashboard(file_path: str, max_blocks: int = 100000) -> go.Figure:
    """Create an interactive dashboard with real-time controls"""
    visualizer = OptimizedCap3DVisualizer(max_blocks_display=max_blocks)
    visualizer.load_data(file_path)
    return visualizer.create_interactive_dashboard(use_batched=True)



if __name__ == "__main__":
    import os
    # Change this to your CAP3D file name
    CAP3D_FILE = "0_120_38_30_89_MET1.cap3d"
    # Get the directory containing this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Look for the file in examples directory (one level up from src)
    cap3d_file = os.path.join(script_dir, "..", "examples", CAP3D_FILE)

    print("=== Testing Enhanced Plotly with Batched Rendering ===")
    print("This should handle 50k+ blocks smoothly!")
    
    # Test new batched rendering (should be MUCH faster)
    print("\n1. Testing batched rendering...")
    fig_batched = quick_preview(cap3d_file, max_blocks=50000)
    print("✓ Batched rendering complete!")
    
    # Test interactive dashboard
    print("\n2. Creating interactive dashboard...")  
    fig_interactive = create_interactive_dashboard(cap3d_file, max_blocks=50000)
    print("✓ Interactive dashboard ready!")
    
    # Show comparison
    print("\n3. Performance comparison available:")
    print("   - Batched: 50k blocks → 2 traces → Fast interaction")
    print("   - Legacy:  2k blocks → 2k traces → Slow interaction")
    
    # Show the visualization
    fig_interactive.show() 
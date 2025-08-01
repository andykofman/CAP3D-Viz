"""
Visualizer Module for CAP3D Enhanced Parser

This module contains the visualization logic for CAP3D data, including
3D rendering, caching, and interactive features.
"""

import time
from itertools import cycle
from typing import List, Tuple, Optional

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

from .data_models import (
    Block, CachedMesh, PolyElement, Layer, Window, Task, ParsedCap3DData
)
from .parser import StreamingCap3DParser


class OptimizedCap3DVisualizer:
    """High-performance 3D visualizer with LOD and caching"""
    
    def __init__(self, max_blocks_display: int = 20000):
        self.max_blocks_display = max_blocks_display
        self.blocks = []
        self.poly_elements = []
        self.layers = []
        self.parse_stats = {}
        self.window = None
        self.task = None
        self.bounds = None
        
        # Caching system for performance
        self._cached_meshes: List[CachedMesh] = []
        self._mesh_cache_valid = False
        self._figure_cache: Optional[go.Figure] = None
        
        # Enhanced color schemes with better contrast
        self.medium_colors = [
            'rgba(31,119,180,0.3)', 'rgba(44,160,44,0.3)', 'rgba(255,127,14,0.3)', 
            'rgba(148,103,189,0.3)', 'rgba(140,86,75,0.3)', 'rgba(227,119,194,0.3)'
        ]
        self.conductor_colors = [
            'rgba(214,39,40,0.9)', 'rgba(227,119,194,0.9)', 'rgba(127,127,127,0.9)', 
            'rgba(188,189,34,0.9)', 'rgba(23,190,207,0.9)', 'rgba(255,20,147,0.9)'
        ]
        self.poly_colors = [
            'rgba(255,165,0,0.6)', 'rgba(147,112,219,0.6)', 'rgba(50,205,50,0.6)',
            'rgba(255,69,0,0.6)', 'rgba(30,144,255,0.6)', 'rgba(255,20,147,0.6)'
        ]
    
    def load_data(self, file_path: str, progress_callback=None):
        """Load complete CAP3D data with enhanced parsing"""
        parser = StreamingCap3DParser(file_path)
        print(f"Loading {file_path} with enhanced parser...")

        start_time = time.time()
        
        # Use comprehensive parser
        parsed_data = parser.parse_complete()
        
        # Store all parsed data
        self.blocks = parsed_data.blocks
        self.poly_elements = parsed_data.poly_elements
        self.layers = parsed_data.layers
        self.window = parsed_data.window
        self.task = parsed_data.task
        self.parse_stats = parsed_data.stats
        
        load_time = time.time() - start_time
        print(f"Loaded enhanced CAP3D data in {load_time:.2f}s")
        
        # Print comprehensive statistics
        print(f"Enhanced parser stats:")
        print(f"  - Blocks: {len(self.blocks)} ({self.parse_stats.get('mediums', 0)} mediums, {self.parse_stats.get('conductors', 0)} conductors)")
        print(f"  - Poly elements: {len(self.poly_elements)}")
        print(f"  - Layers: {len(self.layers)}")
        print(f"  - Window: {'Yes' if self.window else 'No'}")
        print(f"  - Task info: {'Yes' if self.task else 'No'}")
        
        if self.layers:
            layer_types = {}
            for layer in self.layers:
                layer_types[layer.type] = layer_types.get(layer.type, 0) + 1
            print(f"  - Layer types: {dict(layer_types)}")
        
        if self.task and self.task.capacitance_targets:
            print(f"  - Capacitance targets: {self.task.capacitance_targets}")

        # Calculate global bounds including poly elements
        self._calculate_bounds()
        
        # Invalidate caches
        self._mesh_cache_valid = False
        self._figure_cache = None
        
    def _calculate_bounds(self):
        """Calculate bounding box for all blocks and poly elements"""
        if not self.blocks and not self.poly_elements:
            return
            
        all_mins = []
        all_maxs = []

        # Sample blocks for bounds calculation
        sample_blocks = self.blocks[:1000] if len(self.blocks) > 1000 else self.blocks
        for block in sample_blocks:
            min_bound, max_bound = block.bounds
            all_maxs.append(max_bound)
            all_mins.append(min_bound)

        # Include poly elements in bounds calculation
        sample_polys = self.poly_elements[:1000] if len(self.poly_elements) > 1000 else self.poly_elements
        for poly in sample_polys:
            min_bound, max_bound = poly.bounds
            all_maxs.append(max_bound)
            all_mins.append(min_bound)

        if all_mins and all_maxs:
            all_mins = np.array(all_mins)
            all_maxs = np.array(all_maxs)
            self.bounds = (all_mins.min(axis=0), all_maxs.max(axis=0))
            print(f"Global bounds (blocks + polys): {self.bounds}")
        else:
            print("Warning: No valid geometry found for bounds calculation")

    def _generate_poly_mesh(self, poly: PolyElement) -> Optional[CachedMesh]:
        """Generate mesh for a poly element with custom coordinate shape"""
        try:
            # If no custom coordinates, fall back to box shape
            if not poly.coordinates:
                # Use the same logic as blocks for simple case
                vertices = np.array([
                    poly.base,
                    poly.base + poly.v1,
                    poly.base + poly.v1 + poly.v2,
                    poly.base + poly.v2,
                    poly.base + poly.hvector,
                    poly.base + poly.v1 + poly.hvector,
                    poly.base + poly.v1 + poly.v2 + poly.hvector,
                    poly.base + poly.v2 + poly.hvector
                ], dtype=np.float32)
                
                # Standard box faces
                faces = [
                    [0, 1, 2], [0, 2, 3],  # bottom
                    [4, 7, 6], [4, 6, 5],  # top
                    [0, 4, 5], [0, 5, 1],  # front
                    [2, 6, 7], [2, 7, 3],  # back
                    [1, 5, 6], [1, 6, 2],  # right
                    [0, 3, 7], [0, 7, 4],  # left
                ]
            else:
                # Generate mesh from custom coordinates
                # Convert 2D coordinates to 3D vertices by extruding
                bottom_vertices = []
                top_vertices = []
                
                for coord_x, coord_y in poly.coordinates:
                    # Transform 2D coordinates to 3D space
                    world_pos = poly.base + coord_x * poly.v1 + coord_y * poly.v2
                    bottom_vertices.append(world_pos)
                    top_vertices.append(world_pos + poly.hvector)
                
                vertices = np.array(bottom_vertices + top_vertices, dtype=np.float32)
                
                # Generate faces for the polygon
                faces = []
                n_coords = len(poly.coordinates)
                
                # Bottom face (triangulate polygon)
                if n_coords >= 3:
                    for i in range(1, n_coords - 1):
                        faces.append([0, i, i + 1])
                
                # Top face (triangulate polygon, reverse winding)
                if n_coords >= 3:
                    for i in range(1, n_coords - 1):
                        faces.append([n_coords, n_coords + i + 1, n_coords + i])
                
                # Side faces
                for i in range(n_coords):
                    next_i = (i + 1) % n_coords
                    # Two triangles per side
                    faces.append([i, next_i, n_coords + next_i])
                    faces.append([i, n_coords + next_i, n_coords + i])
            
            # Convert faces to plotly format
            i, j, k = [], [], []
            for face in faces:
                i.append(face[0])
                j.append(face[1])
                k.append(face[2])
            
            return CachedMesh(
                x=vertices[:, 0],
                y=vertices[:, 1],
                z=vertices[:, 2],
                i=i,
                j=j,
                k=k,
                block_type='poly',
                block_index=-1,  # Special marker for poly elements
                center=poly.center,
                volume=poly.volume,
                bounds=poly.bounds
            )
            
        except Exception as e:
            print(f"Warning: Failed to generate poly mesh for {poly.name}: {e}")
            return None

    def _build_mesh_cache(self, max_blocks: Optional[int] = None, use_lod: bool = True):
        """Build cached mesh data for all blocks and poly elements"""
        print("Building enhanced mesh cache...")
        start_time = time.time()
        
        # Apply LOD if needed
        blocks_to_cache = self.blocks
        if use_lod and max_blocks and len(blocks_to_cache) > max_blocks:
            blocks_to_cache = self._apply_lod(blocks_to_cache, max_blocks)
        
        self._cached_meshes = []
        total_elements = len(blocks_to_cache) + len(self.poly_elements)
        current_count = 0
        
        # Cache block meshes
        for idx, block in enumerate(blocks_to_cache):
            if current_count % 1000 == 0 and current_count > 0:
                print(f"  Cached {current_count}/{total_elements} meshes...")
                
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
            current_count += 1
        
        # Cache poly element meshes
        for poly_idx, poly in enumerate(self.poly_elements):
            if current_count % 1000 == 0 and current_count > 0:
                print(f"  Cached {current_count}/{total_elements} meshes...")
            
            poly_mesh = self._generate_poly_mesh(poly)
            if poly_mesh:
                # Update the block_index to be unique for poly elements
                poly_mesh.block_index = poly_idx
                self._cached_meshes.append(poly_mesh)
            current_count += 1
        
        self._mesh_cache_valid = True
        cache_time = time.time() - start_time
        print(f"Built enhanced mesh cache for {len(self._cached_meshes)} elements "
              f"({len(blocks_to_cache)} blocks + {len(self.poly_elements)} polys) in {cache_time:.2f}s")

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
        poly_colors = cycle(self.poly_colors)
        
        # Add all traces at once
        for mesh in self._cached_meshes:
            if mesh.block_type == 'medium':
                color = next(medium_colors)
                opacity = 0.3
                name = f"Medium {mesh.block_index}"
            elif mesh.block_type == 'poly':
                color = next(poly_colors)
                opacity = 0.6
                name = f"Poly {mesh.block_index}"
            else:  # conductor
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
                     volume_threshold: Optional[float] = None) -> List[Block]:
        """Advanced filtering with multiple criteria"""
        filtered = []

        for block in self.blocks:
            # Type filter
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
                                          show_polys: bool = True,
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
        poly_colors = cycle(self.poly_colors)
        visible_count = 0
        
        for mesh in self._cached_meshes:
            # Apply filters
            include = True
            
            # Type filter
            if mesh.block_type == 'medium' and not show_mediums:
                include = False
            elif mesh.block_type == 'conductor' and not show_conductors:
                include = False
            elif mesh.block_type == 'poly' and not show_polys:
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
                elif mesh.block_type == 'poly':
                    color = next(poly_colors)
                    opacity = 0.6
                    name = f"Poly {mesh.block_index}"
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
                               show_polys: bool = True,
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
            elif mesh.block_type == 'poly' and not show_polys:
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

    def _add_window_boundaries(self, fig: go.Figure) -> go.Figure:
        """Add window boundaries visualization as wireframe"""
        if not self.window:
            return fig
        
        # Create wireframe box from window corners
        v1, v2 = self.window.v1, self.window.v2
        
        # Create the 8 corners of the window box
        corners = np.array([
            [v1[0], v1[1], v1[2]],  # min corner
            [v2[0], v1[1], v1[2]],  
            [v2[0], v2[1], v1[2]],  
            [v1[0], v2[1], v1[2]],  
            [v1[0], v1[1], v2[2]],  
            [v2[0], v1[1], v2[2]],  
            [v2[0], v2[1], v2[2]],  # max corner
            [v1[0], v2[1], v2[2]]   
        ])
        
        # Define the 12 edges of the box
        edges = [
            [0,1], [1,2], [2,3], [3,0],  # bottom face
            [4,5], [5,6], [6,7], [7,4],  # top face  
            [0,4], [1,5], [2,6], [3,7]   # vertical edges
        ]
        
        # Create wireframe lines
        for edge in edges:
            start, end = edge
            fig.add_trace(go.Scatter3d(
                x=[corners[start][0], corners[end][0], None],
                y=[corners[start][1], corners[end][1], None], 
                z=[corners[start][2], corners[end][2], None],
                mode='lines',
                line=dict(color='yellow', width=4),
                name=f'Window: {self.window.name}' if hasattr(self.window, 'name') and self.window.name else 'Window Boundary',
                showlegend=(edge == edges[0]),  # Only show legend for first edge
                hovertemplate=f'<b>Window Boundary</b><br>X: %{{x}}<br>Y: %{{y}}<br>Z: %{{z}}<extra></extra>'
            ))
        
        return fig

    def create_optimized_visualization(self, 
                                       show_mediums: bool = True,
                                       show_conductors: bool = True,
                                       show_polys: bool = True,
                                       show_window: bool = False,
                                       z_slice: Optional[float] = None,                                     
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
            fig = self._create_batched_visualization(
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
                show_polys=show_polys,
                z_max=z_slice
            )
        else:
            # Fallback to old method (for comparison)
            print("Using non-cached visualization (slower)...")
            fig = self._create_visualization_legacy(
                show_mediums, show_conductors, z_slice, max_blocks, use_lod, show_edges, 
                opacity_mediums, opacity_conductors
            )
        
        # Add window boundaries if requested
        if show_window:
            fig = self._add_window_boundaries(fig)
        
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
            show_mediums=show_mediums,
            show_conductors=show_conductors,
            z_max=z_slice
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
        """Apply Level of Detail - prioritize larger, more important blocks"""
        if len(blocks) <= max_blocks:
            return blocks
        # Sort by volume (Larger blocks are more important)
        blocks_with_priority = [(block, block.volume) for block in blocks]
        blocks_with_priority.sort(key=lambda x: x[1], reverse=True)

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
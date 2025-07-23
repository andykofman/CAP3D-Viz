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
from typing import Generator, Dict, List, Text, Tuple, Optional
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
    """ Memory-efficient straming parser for large cap3d files """
    def __init__(self, file_path:str):
        self.file_path = file_path
        self.stats     = {
            'total_blocks' : 0,
            'conductors'   : 0,
            'mediums'      : 0,
            'parse_time'   : 0

      }

    def parse_blocks_straming(self, chunk_size: int = 1024 *1024) -> Generator[Block, None, None]:
        """Stream parse block without loading entire file into memory"""
        start_time = time.time()

        with open(self.file_path, '+r') as f:
            content = f.read() # for now, read all content to handle format differences


        # Handle both formats: </tag> and [/tag]
        # First, normalize the format
        content = re.sub(r'\[/(\w+)\]', r'</\1>', content)

        # parse mediums
        """   find all <medium>...</medium> sections (or until the next block tag or end of file)."""
        for medium_match in re.finditer(r'<medium>(.*?)(?=</medium>|<(?:medium|conductor|window|task|cap3d)>|$)', content, re.DOTALL):
            section_content = medium_match.group(1)
            yield from self._parse_section('medium', section_content)

        # parse conductors
        for conductor_match in re.finditer(r'<conductor>(.*?)(?=</conductor>|<(?:medium|conductor|window|task|cap3d)>|$)', content, re.DOTALL):
            section_content = conductor_match.group(1)
            yield from self._parse_section('conductor', section_content)
        
        self.stats['parse_time'] = time.time() - start_time

    def _parse_section(self, section_type:str, content:str) -> Generator[Block, None, None]:
            """ Parse a complete medium or conductor section """

            # Extract section name and properties
            name_match = re.search(r'name\s+(\S+)', content)
            name = name_match.group(1) if name_match else f"unname_{section_type}"

            diel = None
            if section_type == 'medium':
                diel_match =  re.search(r'diel\s+([\d\.]+)', content)
                diel = float(diel_match.group(1)) if diel_match else None
                self.stats['mediums'] = self.stats['mediums'] + 1
            else:
                self.stats['conductors'] = self.stats['conductors'] + 1
            
            # Extract all blocks in this section
            for block_match in re.finditer(r'<block>(.*?)</block>', content, re.DOTALL):
                block_content = block_match.group(1)
            
                # Extract block properties with error handling
                try: 
                    block_name_match = re.search(r'name\s+(\S+)', block_content)
                    block_name = block_name_match.group(1) if block_name_match else f"block_{self.stats['total_blocks']}"

                    base_match = re.search(r'basepoint\s*\((.*?)\)', block_content)
                    v1_match = re.search(r'v1\s*\((.*?)\)', block_content)
                    v2_match = re.search(r'v2\s*\((.*?)\)', block_content)
                    hvec_match = re.search(r'hvector\s*\((.*?)\)', block_content)
                    
                    if not (base_match and v1_match and v2_match and hvec_match):
                        continue

                    base = [float(x.strip()) for x in base_match.group(1).split(',')]
                    v1 = [float(x.strip()) for x in v1_match.group(1).split(',')]
                    v2 = [float(x.strip()) for x in v2_match.group(1).split(',')]
                    hvec = [float(x.strip()) for x in hvec_match.group(1).split(',')]

                    block = Block(
                        name=block_name,
                        type=section_type,
                        parent_name=name,
                        base=base,
                        v1=v1,
                        v2=v2,
                        hvec=hvec,
                        diel=diel
                    )
                
                    self.stats['total_blocks'] = self.stats['total_blocks'] +1 
                    yield block
                except(ValueError, AttributeError) as e:
                    
                    print(f"Warning: failed to parse block in {name}: {e}")
                    continue

            
# color schemes
from itertools import cycle

class OptimizedCap3DVisualizer:
    """ High-performance 3D visualizer with LOD and filtering """
    def __init__ (self, max_blocks_display: int = 20000):
        self.max_blocks_display = max_blocks_display
        self.blocks = []
        self.bounds = None

        # Enhanced color schemes with better contrast
        self.medium_colors = cycle(['rgba(31,119,180,0.3)', 'rgba(44,160,44,0.3)', 'rgba(255,127,14,0.3)', 
                            'rgba(148,103,189,0.3)', 'rgba(140,86,75,0.3)', 'rgba(227,119,194,0.3)'])
        self.conductor_colors = cycle(['rgba(214,39,40,0.9)', 'rgba(227,119,194,0.9)', 'rgba(127,127,127,0.9)', 
                                'rgba(188,189,34,0.9)', 'rgba(23,190,207,0.9)', 'rgba(255,20,147,0.9)'])

    def load_data(self, file_path:str, progress_callback: None):
        """  Load data with progress tracking """
        parser = StreamingCap3DParser(file_path)
        print(f"Loading {file_path}...")

        start_time = time.time()
        self.blocks = []

        for i, block, in enumerate(parser.parse_blocks_straming()):
            self.blocks.append(block)

            if progress_callback and i % 1000 ==0:
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

        for block in self.block[:1000]: # Sample first 1000 blocks for bounds 
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

        fig = go.figure()


                





# if __name__ == "__main__":
#     # Basic manual test for Block
#     base = [0, 0, 0]
#     v1 = [1, 0, 0]
#     v2 = [0, 1, 0]
#     hvec = [0, 0, 1]
#     block = Block("test_block", "medium", "parent", base, v1, v2, hvec, diel=2.5)
#     print("Block name:", block.name)
#     print("Block vertices:\n", block.vertices)
#     print("Block bounds:", block.bounds)
#     print("Block volume:", block.volume)
#     print("Block center:", block.center)

#     # StreamingCap3DParser test would require a file, so not run here
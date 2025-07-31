"""
CAP3D Enhanced Parser - Optimized for Large Files

This module provides enhanced parsing and visualization capabilities
for large CAP3D files that may be too memory-intensive for the standard parser.

"""

import re
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
        
@dataclass
class Layer: 
    """ Layer definition with name and type """
    name: str
    type: str # interconnect, via, etc.
    
@dataclass
class PolyElement:
    """ Polygonal element with custom geometry """
    name: str
    parent_name: str
    base: np.ndarray
    v1: np.ndarray
    v2: np.ndarray
    hvector: np.ndarray
    coordinates: List[Tuple[float, float]] # from <coord> tag

    @property
    def bounds(self) -> Tuple[np.ndarray,np.ndarray]:
        """ Get bounding box for polygon"""
        # Calculate bounds from base vectors
        base_bounds = np.array([self.base,
                                self.base + self.v1,
                                self.base + self.v2,
                                self.base + self.hvector,
                                self.base + self.v1 +self.v2,
                                self.base + self.v1 + self.hvector,
                                self.base + self.v2 + self.hvector,
                                self.base + self.v2 + self.v1 + self.hvector])
        
        return base_bounds.min(axis=0), base_bounds.max(axis=0)

    
    @property		
    def center(self) -> np.ndarray:		

        """Get center point for LOD calculations"""		
        return self.base + 0.5 * (self.v1 + self.v2 + self.hvector)		
                
    @property		
    def volume(self) -> float:		
        """Calculate volume for LOD prioritization"""		
        return float(abs(np.dot(self.v1, np.cross(self.v2, self.hvector))))


 
@dataclass
class Window: 
    """ Simulation window/boundary definition"""
    name: Optional [str]
    v1: np.ndarray # Corner 1
    v2: np.ndarray # Corner 2
    boundary_type: Optional[str] = None # e.g., 'dirchilet'


@dataclass
class Task:
    """ Simulation task definition"""
    capacitance_targets: List[str] # List of conductor names for capacitance calculations


@dataclass
class ParsedCap3DData:
    """ Complete parsed CAP3D data structure"""
    blocks: List[Block]
    poly_elements: List[PolyElement]
    layers: List[Layer]
    window: Optional [Window]
    task: Optional[Task]
    stats: Dict


class ParserState:
    """Optimized parser state for state-machine based parsing"""
    def __init__(self):
        # Section state
        self.current_section = None
        self.current_section_name = None
        self.current_diel = None
        
        # Context flags
        self.in_block = False
        self.in_poly = False
        self.in_task = False
        self.in_capacitance = False
        
        # Data containers
        self.block_data = {}
        self.poly_data = {}
        self.layer_data = {}
        self.window_data = {}
        self.task_data = {'capacitance_targets': []}
        self.coord_buffer = []
        
        # Pending objects for efficient collection
        self.pending_block = None
        self.pending_poly = None
        self.pending_layer = None
        self.pending_window = None
        self.pending_task = None


class StreamingCap3DParser:
    """ Memory-efficient streaming parser for large cap3d files """
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.stats = {
            'total_blocks': 0,
            'conductors': 0,
            'mediums': 0,
            'poly_elements': 0,  # new
            'layers':0,          # new
            'has_window': False, # new
            'has_task': False,   # new
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

    def parse_complete(self) -> ParsedCap3DData:
        """Optimized comprehensive parser using state machine for better performance"""
        start_time = time.time()
        
        # Initialize data containers
        blocks, poly_elements, layers = [], [], []
        window, task = None, None
        
        # Pre-compile common patterns for performance
        LAYER_TYPES = {'interconnect', 'via', 'metal', 'poly', 'contact'}
        BOUNDARY_TYPES = {'dirichlet', 'neumann'}
        
        # State-based dispatch tables for efficient parsing
        tag_handlers = self._create_tag_handlers()
        property_handlers = self._create_property_handlers()
        
        with open(self.file_path, 'r', encoding='utf-8', buffering=8192) as f:
            # Parser state
            state = ParserState()
            
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line or line.startswith('<!--'):
                    continue
                
                # Fast path: Use state-based dispatch instead of checking all conditions
                self._handle_line_optimized(line, state, tag_handlers, property_handlers, 
                                           blocks, poly_elements, layers, LAYER_TYPES, BOUNDARY_TYPES)
                
                # Collect completed objects efficiently
                if state.pending_block:
                    blocks.append(state.pending_block)
                    state.pending_block = None
                
                if state.pending_poly:
                    poly_elements.append(state.pending_poly)
                    state.pending_poly = None
                
                if state.pending_layer:
                    layers.append(state.pending_layer)
                    state.pending_layer = None
                    
                if state.pending_window:
                    window = state.pending_window
                    state.pending_window = None
                    self.stats['has_window'] = True
                
                if state.pending_task:
                    task = state.pending_task
                    state.pending_task = None
                    self.stats['has_task'] = True
        
        self.stats['parse_time'] = time.time() - start_time
        
        return ParsedCap3DData(
            blocks=blocks,
            poly_elements=poly_elements,
            layers=layers,
            window=window,
            task=task,
            stats=self.stats.copy()
        )
    
    def _create_tag_handlers(self):
        """Create optimized tag dispatch table"""
        return {
            '<layer>': self._start_layer,
            '</layer>': self._end_layer,
            '<window>': self._start_window,
            '</window>': self._end_window,
            '<task>': self._start_task,
            '</task>': self._end_task,
            '<medium>': self._start_medium,
            '<conductor>': self._start_conductor,
            '</medium>': self._end_section,
            '</conductor>': self._end_section,
            '<block>': self._start_block,
            '</block>': self._end_block,
            '<poly>': self._start_poly,
            '</poly>': self._end_poly,
            '<coord>': self._start_coord,
            '</coord>': self._end_coord,
            '<capacitance': self._start_capacitance,
            '</capacitance': self._end_capacitance,
        }
    
    def _create_property_handlers(self):
        """Create optimized property dispatch table"""
        return {
            'name ': self._handle_name,
            'type ': self._handle_type,
            'diel ': self._handle_diel,
            'basepoint(': self._handle_basepoint,
            'v1(': self._handle_v1,
            'v2(': self._handle_v2,
            'hvector(': self._handle_hvector,
        }
    
    def _handle_line_optimized(self, line, state, tag_handlers, property_handlers, 
                              blocks, poly_elements, layers, layer_types, boundary_types):
        """Optimized line handler using state-based dispatch"""
        
        # Fast path 1: Check for exact tag matches first
        if line in tag_handlers:
            tag_handlers[line](state)
            return True
        
        # Fast path 2: Check for tag prefixes (for tags with attributes)
        for tag_prefix in ['<capacitance', '</capacitance']:
            if line.startswith(tag_prefix):
                tag_handlers[tag_prefix](state)
                return True
        
        # Fast path 3: State-based property handling (most efficient)
        # Check in_block and in_poly FIRST before checking current_section
        if state.in_block:
            return self._handle_block_properties(line, state, property_handlers)
        elif state.in_poly:
            return self._handle_poly_properties(line, state, property_handlers)
        elif state.in_capacitance:
            return self._handle_capacitance_properties(line, state)
        elif state.current_section == 'layer':
            return self._handle_layer_properties(line, state, layer_types)
        elif state.current_section == 'window':
            return self._handle_window_properties(line, state, property_handlers, boundary_types)
        elif state.current_section in ['medium', 'conductor']:
            return self._handle_section_properties(line, state, property_handlers)
        
        # Handle coordinate data in poly context (should be handled in _handle_poly_properties)
        # This is a fallback for any missed cases
        if state.in_poly and not line.startswith('<') and not line.startswith('name') and not line.startswith('basepoint'):
            state.coord_buffer.extend(self._parse_coordinate_pairs(line))
            return True
        
        return False
    
    # Optimized tag handlers
    def _start_layer(self, state): 
        state.current_section = 'layer'
        state.layer_data = {}
    
    def _end_layer(self, state):
        if state.layer_data.get('name') and state.layer_data.get('type'):
            # Direct append without function call overhead
            state.pending_layer = Layer(
                name=state.layer_data['name'],
                type=state.layer_data['type']
            )
            self.stats['layers'] += 1
        state.current_section = None
        state.layer_data = {}
    
    def _start_window(self, state):
        state.current_section = 'window'
        state.window_data = {}
    
    def _end_window(self, state):
        if state.window_data.get('v1') and state.window_data.get('v2'):
            state.pending_window = Window(
                name=state.window_data.get('name'),
                v1=np.array(state.window_data['v1'], dtype=np.float32),
                v2=np.array(state.window_data['v2'], dtype=np.float32),
                boundary_type=state.window_data.get('boundary_type')
            )
        state.current_section = None
        state.window_data = {}
    
    def _start_task(self, state):
        state.current_section = 'task'
        state.in_task = True
        state.task_data = {'capacitance_targets': []}
    
    def _end_task(self, state):
        if state.task_data['capacitance_targets']:
            state.pending_task = Task(capacitance_targets=state.task_data['capacitance_targets'])
        state.current_section = None
        state.in_task = False
        state.task_data = {'capacitance_targets': []}
    
    def _start_medium(self, state):
        state.current_section = 'medium'
        state.current_section_name = None
        state.current_diel = None
    
    def _start_conductor(self, state):
        state.current_section = 'conductor'
        state.current_section_name = None
        state.current_diel = None
    
    def _end_section(self, state):
        state.current_section = None
        state.current_section_name = None
        state.current_diel = None
    
    def _start_block(self, state):
        state.in_block = True
        state.block_data = {
            'section_type': state.current_section,
            'section_name': state.current_section_name,
            'diel': state.current_diel
        }
    
    def _end_block(self, state):
        if state.in_block:
            state.in_block = False
            if self._is_valid_block(state.block_data):
                block = self._create_block(state.block_data)
                if block:
                    state.pending_block = block
                    self.stats['total_blocks'] += 1
                    if block.type == 'medium':
                        self.stats['mediums'] += 1
                    else:
                        self.stats['conductors'] += 1
            state.block_data = {}
    
    def _start_poly(self, state):
        state.in_poly = True
        state.poly_data = {
            'section_type': state.current_section,
            'section_name': state.current_section_name
        }
        state.coord_buffer = []
    
    def _end_poly(self, state):
        state.in_poly = False
        if self._is_valid_poly(state.poly_data):
            poly_element = self._create_poly_element(state.poly_data, state.coord_buffer)
            if poly_element:
                state.pending_poly = poly_element
                self.stats['poly_elements'] += 1
        state.poly_data = {}
        state.coord_buffer = []
    
    def _start_coord(self, state):
        pass  # Coordinate handling is done in property parsing
    
    def _end_coord(self, state):
        pass  # End of coord section
    
    def _start_capacitance(self, state):
        if state.in_task:
            state.in_capacitance = True
    
    def _end_capacitance(self, state):
        if state.in_task:
            state.in_capacitance = False
    
    # Optimized property handlers with reduced string operations
    def _handle_block_properties(self, line, state, property_handlers):
        """Handle block properties efficiently"""
        # Quick check for common property prefixes
        if line.startswith('name '):
            state.block_data['name'] = line[5:].strip()
            return True
        elif line.startswith('basepoint(') and line.endswith(')'):
            coords_str = line[10:-1]
            state.block_data['base'] = self._parse_coords(coords_str)
            return True
        elif line.startswith('v1(') and line.endswith(')'):
            coords_str = line[3:-1]
            state.block_data['v1'] = self._parse_coords(coords_str)
            return True
        elif line.startswith('v2(') and line.endswith(')'):
            coords_str = line[3:-1]
            state.block_data['v2'] = self._parse_coords(coords_str)
            return True
        elif line.startswith('hvector(') and line.endswith(')'):
            coords_str = line[8:-1]
            state.block_data['hvec'] = self._parse_coords(coords_str)
            return True
        return False
    
    def _handle_poly_properties(self, line, state, property_handlers):
        """Handle poly properties efficiently"""
        if line.startswith('<coord>'):
            # Extract coordinate from the coord line
            coord_text = line[7:]  # Remove '<coord>'
            if coord_text.endswith('</coord>'):
                coord_text = coord_text[:-8]  # Remove '</coord>'
            state.coord_buffer.extend(self._parse_coordinate_pairs(coord_text))
            return True
        elif line.startswith('</coord>'):
            return True  # End of coord section
        elif line.startswith('name '):
            state.poly_data['name'] = line[5:].strip()
            return True
        elif line.startswith('basepoint(') and line.endswith(')'):
            coords_str = line[10:-1]
            state.poly_data['base'] = self._parse_coords(coords_str)
            return True
        elif line.startswith('v1(') and line.endswith(')'):
            coords_str = line[3:-1]
            state.poly_data['v1'] = self._parse_coords(coords_str)
            return True
        elif line.startswith('v2(') and line.endswith(')'):
            coords_str = line[3:-1]
            state.poly_data['v2'] = self._parse_coords(coords_str)
            return True
        elif line.startswith('hvector(') and line.endswith(')'):
            coords_str = line[8:-1]
            state.poly_data['hvec'] = self._parse_coords(coords_str)
            return True
        elif not line.startswith('<') and not line.startswith('name') and not line.startswith('basepoint'):
            # Multi-line coordinate data
            state.coord_buffer.extend(self._parse_coordinate_pairs(line))
            return True
        return False
    
    def _handle_layer_properties(self, line, state, layer_types):
        """Handle layer properties efficiently"""
        if line.startswith('name '):
            state.layer_data['name'] = line[5:].strip()
        elif line.startswith('type '):
            state.layer_data['type'] = line[5:].strip()
        elif line in layer_types:
            state.layer_data['type'] = line
        else:
            return False
        return True
    
    def _handle_window_properties(self, line, state, property_handlers, boundary_types):
        """Handle window properties efficiently"""
        if line.startswith('name '):
            state.window_data['name'] = line[5:].strip()
        elif line in boundary_types:
            state.window_data['boundary_type'] = line
        else:
            for prefix, handler in property_handlers.items():
                if line.startswith(prefix):
                    handler(line, state.window_data, prefix)
                    return True
        return True
    
    def _handle_capacitance_properties(self, line, state):
        """Handle capacitance properties efficiently"""
        if not line.startswith('<') and not line.startswith('</'):
            conductor_name = line.strip()
            if conductor_name:
                state.task_data['capacitance_targets'].append(conductor_name)
        return True
    
    def _handle_section_properties(self, line, state, property_handlers):
        """Handle medium/conductor section properties efficiently"""
        if line.startswith('name '):
            state.current_section_name = line[5:].strip()
        elif state.current_section == 'medium' and line.startswith('diel '):
            state.current_diel = float(line[5:].strip())
        else:
            return False
        return True
    
    # Optimized property parsers
    def _handle_name(self, line, data_dict, prefix):
        data_dict['name'] = line[len(prefix):].strip()
    
    def _handle_type(self, line, data_dict, prefix):
        data_dict['type'] = line[len(prefix):].strip()
    
    def _handle_diel(self, line, data_dict, prefix):
        data_dict['diel'] = float(line[len(prefix):].strip())
    
    def _handle_basepoint(self, line, data_dict, prefix):
        if line.endswith(')'):
            coords_str = line[len(prefix):-1]
            data_dict['base'] = self._parse_coords(coords_str)
    
    def _handle_v1(self, line, data_dict, prefix):
        if line.endswith(')'):
            coords_str = line[len(prefix):-1]
            data_dict['v1'] = self._parse_coords(coords_str)
    
    def _handle_v2(self, line, data_dict, prefix):
        if line.endswith(')'):
            coords_str = line[len(prefix):-1]
            data_dict['v2'] = self._parse_coords(coords_str)
    
    def _handle_hvector(self, line, data_dict, prefix):
        if line.endswith(')'):
            coords_str = line[len(prefix):-1]
            data_dict['hvec'] = self._parse_coords(coords_str)
    


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

    def _is_valid_poly(self, poly_data: dict) -> bool:
        """Check if poly element has all required fields"""
        required = ['section_type', 'section_name', 'base', 'v1', 'v2', 'hvec']
        return all(key in poly_data for key in required)

    def _create_poly_element(self, poly_data: dict, coordinates: List[Tuple[float, float]]) -> Optional[PolyElement]:
        """Create PolyElement object from parsed data"""
        try:
            return PolyElement(
                name=poly_data.get('name', f"poly_{self.stats['poly_elements']}"),
                parent_name=poly_data['section_name'],
                base=np.array(poly_data['base'], dtype=np.float32),
                v1=np.array(poly_data['v1'], dtype=np.float32),
                v2=np.array(poly_data['v2'], dtype=np.float32),
                hvector=np.array(poly_data['hvec'], dtype=np.float32),
                coordinates=coordinates
            )
        except (ValueError, KeyError) as e:
            print(f"Warning: failed to create poly element: {e}")
            return None

    def _parse_coordinate_pairs(self, coord_text: str) -> List[Tuple[float, float]]:
        """Parse coordinate pairs from text like '(1.0,2.0) (3.0,4.0)'"""
        coordinates = []
        try:
            # Remove extra whitespace and split by closing parenthesis
            coord_text = coord_text.strip()
            if not coord_text:
                return coordinates
            
            # Find all coordinate pairs using simple parsing
            import re
            # Match patterns like (x,y)
            pattern = r'\(([^)]+)\)'
            matches = re.findall(pattern, coord_text)
            
            for match in matches:
                # Split by comma and convert to float
                parts = match.split(',')
                if len(parts) == 2:
                    x = float(parts[0].strip())
                    y = float(parts[1].strip())
                    coordinates.append((x, y))
        except (ValueError, IndexError) as e:
            print(f"Warning: failed to parse coordinates '{coord_text}': {e}")
        
        return coordinates

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
        self.blocks        = []
        self.poly_elements = []
        self.layers        = []
        self.parse_stats   = {}
        self.window        = None
        self.task          = None
        self.bounds        = None
        
        # Caching system for performance
        self._cached_meshes: List[CachedMesh] = []
        self._mesh_cache_valid = False
        self._figure_cache: Optional[go.Figure] = None
        
        # Enhanced color schemes with better contrast
        self.medium_colors = ['rgba(31,119,180,0.3)', 'rgba(44,160,44,0.3)', 'rgba(255,127,14,0.3)', 
                            'rgba(148,103,189,0.3)', 'rgba(140,86,75,0.3)', 'rgba(227,119,194,0.3)']
        self.conductor_colors = ['rgba(214,39,40,0.9)', 'rgba(227,119,194,0.9)', 'rgba(127,127,127,0.9)', 
                                'rgba(188,189,34,0.9)', 'rgba(23,190,207,0.9)', 'rgba(255,20,147,0.9)']
        self.poly_colors = ['rgba(255,165,0,0.6)', 'rgba(147,112,219,0.6)', 'rgba(50,205,50,0.6)',
                                'rgba(255,69,0,0.6)', 'rgba(30,144,255,0.6)', 'rgba(255,20,147,0.6)']
    
    
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

        # calculate global bounds including poly elements
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
            else: # conductor
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
"""
Parser Module for CAP3D Enhanced Parser

This module contains the parsing logic for CAP3D files, including
the state machine parser and streaming parser for large files.
"""

import re
import time
from typing import Generator, Dict, List, Tuple, Optional, Set, Callable, Any, MutableMapping
from concurrent.futures import ThreadPoolExecutor

from .data_models import (
    Block, PolyElement, Layer, Window, Task, ParsedCap3DData
)


class ParserState:
    """Optimized parser state for state-machine based parsing"""
    
    # Section state
    current_section: Optional[str]
    current_section_name: Optional[str]
    current_diel: Optional[float]
    
    # Context flags
    in_block: bool
    in_poly: bool
    in_task: bool
    in_capacitance: bool
    
    # Data containers
    block_data: Dict[str, Any]
    poly_data: Dict[str, Any]
    layer_data: Dict[str, Any]
    window_data: Dict[str, Any]
    task_data: Dict[str, Any]
    coord_buffer: List[Tuple[float, float]]
    
    # Pending objects for efficient collection
    pending_block: Optional[Block]
    pending_poly: Optional[PolyElement]
    pending_layer: Optional[Layer]
    pending_window: Optional[Window]
    pending_task: Optional[Task]

    def __init__(self) -> None:
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
    """Memory-efficient streaming parser for large cap3d files"""
    
    def __init__(self, file_path: str) -> None:
        self.file_path = file_path
        self.stats: Dict[str, Any] = {
            'total_blocks': 0,
            'conductors': 0,
            'mediums': 0,
            'poly_elements': 0,
            'layers': 0,
            'has_window': False,
            'has_task': False,
            'parse_time': 0.0,
        }

    def parse_blocks_streaming(self) -> Generator[Block, None, None]:
        """True streaming parser - processes file line by line"""
        start_time = time.time()
        
        with open(self.file_path, 'r', encoding='utf-8', buffering=8192) as f:
            current_section = None  # 'medium' or 'conductor'
            current_section_name = None
            current_diel = None
            in_block = False
            block_data: Dict[str, Any] = {}
            
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
                    block_data = {
                        'section_type': current_section, 
                        'section_name': current_section_name, 
                        'diel': current_diel
                    }
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
        blocks: List[Block] = []
        poly_elements: List[PolyElement] = []
        layers: List[Layer] = []
        window: Optional[Window]
        task: Optional[Task]
        window, task = None, None
        
        # Pre-compile common patterns for performance
        LAYER_TYPES: Set[str] = {'interconnect', 'via', 'metal', 'poly', 'contact'}
        BOUNDARY_TYPES: Set[str] = {'dirichlet', 'neumann'}
        
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
            stats=self.stats.copy(),
        )
    
    def _create_tag_handlers(self) -> Dict[str, Callable[[ParserState], None]]:
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
    
    def _create_property_handlers(self) -> Dict[str, Callable[[str, MutableMapping[str, Any], str], None]]:
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
    
    def _handle_line_optimized(
        self,
        line: str,
        state: ParserState,
        tag_handlers: Dict[str, Callable[[ParserState], None]],
        property_handlers: Dict[str, Callable[[str, MutableMapping[str, Any], str], None]],
        blocks: List[Block],
        poly_elements: List[PolyElement],
        layers: List[Layer],
        layer_types: Set[str],
        boundary_types: Set[str],
    ) -> bool:
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
    def _start_layer(self, state: ParserState) -> None: 
        state.current_section = 'layer'
        state.layer_data = {}
    
    def _end_layer(self, state: ParserState) -> None:
        if state.layer_data.get('name') and state.layer_data.get('type'):
            # Direct append without function call overhead
            state.pending_layer = Layer(
                name=state.layer_data['name'],
                type=state.layer_data['type']
            )
            self.stats['layers'] += 1
        state.current_section = None
        state.layer_data = {}
    
    def _start_window(self, state: ParserState) -> None:
        state.current_section = 'window'
        state.window_data = {}
    
    def _end_window(self, state: ParserState) -> None:
        if state.window_data.get('v1') and state.window_data.get('v2'):
            import numpy as np
            state.pending_window = Window(
                name=state.window_data.get('name'),
                v1=np.array(state.window_data['v1'], dtype=np.float32),
                v2=np.array(state.window_data['v2'], dtype=np.float32),
                boundary_type=state.window_data.get('boundary_type')
            )
        state.current_section = None
        state.window_data = {}
    
    def _start_task(self, state: ParserState) -> None:
        state.current_section = 'task'
        state.in_task = True
        state.task_data = {'capacitance_targets': []}
    
    def _end_task(self, state: ParserState) -> None:
        if state.task_data['capacitance_targets']:
            state.pending_task = Task(capacitance_targets=state.task_data['capacitance_targets'])
        state.current_section = None
        state.in_task = False
        state.task_data = {'capacitance_targets': []}
    
    def _start_medium(self, state: ParserState) -> None:
        state.current_section = 'medium'
        state.current_section_name = None
        state.current_diel = None
    
    def _start_conductor(self, state: ParserState) -> None:
        state.current_section = 'conductor'
        state.current_section_name = None
        state.current_diel = None
    
    def _end_section(self, state: ParserState) -> None:
        state.current_section = None
        state.current_section_name = None
        state.current_diel = None
    
    def _start_block(self, state: ParserState) -> None:
        state.in_block = True
        state.block_data = {
            'section_type': state.current_section,
            'section_name': state.current_section_name,
            'diel': state.current_diel
        }
    
    def _end_block(self, state: ParserState) -> None:
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
    
    def _start_poly(self, state: ParserState) -> None:
        state.in_poly = True
        state.poly_data = {
            'section_type': state.current_section,
            'section_name': state.current_section_name
        }
        state.coord_buffer = []
    
    def _end_poly(self, state: ParserState) -> None:
        state.in_poly = False
        if self._is_valid_poly(state.poly_data):
            poly_element = self._create_poly_element(state.poly_data, state.coord_buffer)
            if poly_element:
                state.pending_poly = poly_element
                self.stats['poly_elements'] += 1
        state.poly_data = {}
        state.coord_buffer = []
    
    def _start_coord(self, state: ParserState) -> None:
        pass  # Coordinate handling is done in property parsing
    
    def _end_coord(self, state: ParserState) -> None:
        pass  # End of coord section
    
    def _start_capacitance(self, state: ParserState) -> None:
        if state.in_task:
            state.in_capacitance = True
    
    def _end_capacitance(self, state: ParserState) -> None:
        if state.in_task:
            state.in_capacitance = False
    
    # Optimized property handlers with reduced string operations
    def _handle_block_properties(
        self,
        line: str,
        state: ParserState,
        property_handlers: Dict[str, Callable[[str, MutableMapping[str, Any], str], None]],
    ) -> bool:
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
    
    def _handle_poly_properties(
        self,
        line: str,
        state: ParserState,
        property_handlers: Dict[str, Callable[[str, MutableMapping[str, Any], str], None]],
    ) -> bool:
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
    
    def _handle_layer_properties(self, line: str, state: ParserState, layer_types: Set[str]) -> bool:
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
    
    def _handle_window_properties(
        self,
        line: str,
        state: ParserState,
        property_handlers: Dict[str, Callable[[str, MutableMapping[str, Any], str], None]],
        boundary_types: Set[str],
    ) -> bool:
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
    
    def _handle_capacitance_properties(self, line: str, state: ParserState) -> bool:
        """Handle capacitance properties efficiently"""
        if not line.startswith('<') and not line.startswith('</'):
            conductor_name = line.strip()
            if conductor_name:
                state.task_data['capacitance_targets'].append(conductor_name)
        return True
    
    def _handle_section_properties(
        self,
        line: str,
        state: ParserState,
        property_handlers: Dict[str, Callable[[str, MutableMapping[str, Any], str], None]],
    ) -> bool:
        """Handle medium/conductor section properties efficiently"""
        if line.startswith('name '):
            state.current_section_name = line[5:].strip()
        elif state.current_section == 'medium' and line.startswith('diel '):
            state.current_diel = float(line[5:].strip())
        else:
            return False
        return True
    
    # Optimized property parsers
    def _handle_name(self, line: str, data_dict: MutableMapping[str, Any], prefix: str) -> None:
        data_dict['name'] = line[len(prefix):].strip()
    
    def _handle_type(self, line: str, data_dict: MutableMapping[str, Any], prefix: str) -> None:
        data_dict['type'] = line[len(prefix):].strip()
    
    def _handle_diel(self, line: str, data_dict: MutableMapping[str, Any], prefix: str) -> None:
        data_dict['diel'] = float(line[len(prefix):].strip())
    
    def _handle_basepoint(self, line: str, data_dict: MutableMapping[str, Any], prefix: str) -> None:
        if line.endswith(')'):
            coords_str = line[len(prefix):-1]
            data_dict['base'] = self._parse_coords(coords_str)
    
    def _handle_v1(self, line: str, data_dict: MutableMapping[str, Any], prefix: str) -> None:
        if line.endswith(')'):
            coords_str = line[len(prefix):-1]
            data_dict['v1'] = self._parse_coords(coords_str)
    
    def _handle_v2(self, line: str, data_dict: MutableMapping[str, Any], prefix: str) -> None:
        if line.endswith(')'):
            coords_str = line[len(prefix):-1]
            data_dict['v2'] = self._parse_coords(coords_str)
    
    def _handle_hvector(self, line: str, data_dict: MutableMapping[str, Any], prefix: str) -> None:
        if line.endswith(')'):
            coords_str = line[len(prefix):-1]
            data_dict['hvec'] = self._parse_coords(coords_str)

    def _parse_coords(self, coords_str: str) -> List[float]:
        """Fast coordinate parsing without regex"""
        try:
            return [float(x.strip()) for x in coords_str.split(',')]
        except ValueError:
            return [0.0, 0.0, 0.0]

    def _is_valid_block(self, block_data: Dict[str, Any]) -> bool:
        """Check if block has all required fields"""
        required = ['section_type', 'section_name', 'base', 'v1', 'v2', 'hvec']
        return all(key in block_data for key in required)

    def _create_block(self, block_data: Dict[str, Any]) -> Optional[Block]:
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

    def _is_valid_poly(self, poly_data: Dict[str, Any]) -> bool:
        """Check if poly element has all required fields"""
        required = ['section_type', 'section_name', 'base', 'v1', 'v2', 'hvec']
        return all(key in poly_data for key in required)

    def _create_poly_element(self, poly_data: Dict[str, Any], coordinates: List[Tuple[float, float]]) -> Optional[PolyElement]:
        """Create PolyElement object from parsed data"""
        try:
            import numpy as np
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
        coordinates: List[Tuple[float, float]] = []
        try:
            # Remove extra whitespace and split by closing parenthesis
            coord_text = coord_text.strip()
            if not coord_text:
                return coordinates
            
            # Find all coordinate pairs using simple parsing
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
"""
Data Models for CAP3D Enhanced Parser

This module contains all the core data structures and models used
throughout the CAP3D enhanced parsing and visualization system.
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict


class Block:
    """Represents a 3D block in the CAP3D file"""

    def __init__(self, name: str, type: str, parent_name: str, 
                 base, v1, v2, hvec, diel: Optional[float] = None):
        
        self.name = name 
        self.type = type  # medium or conductor 
        self.parent_name = parent_name 
        self.diel = diel 

        # Ensure all vectors are numpy arrays
        self.base = np.array(base, dtype=np.float32)
        self.v1   = np.array(v1, dtype=np.float32)
        self.v2   = np.array(v2, dtype=np.float32)
        self.hvec = np.array(hvec, dtype=np.float32)

    @property 
    def vertices(self) -> np.ndarray:
        """Generate 8 vertices of the box efficiently"""
        x, y, z = self.base
        dx, _, _ = self.v1
        _, dy, _ = self.v2
        _, _, dz = self.hvec

        # Vectorized vertex generation
        vertices = np.array([
            [x, y, z], [x+dx, y, z], [x+dx, y+dy, z], [x, y+dy, z],
            [x, y, z+dz], [x+dx, y, z+dz], [x+dx, y+dy, z+dz], [x, y+dy, z+dz]
        ], dtype=np.float32)

        return vertices

    @property 
    def bounds(self) -> Tuple[np.ndarray, np.ndarray]: 
        """Get bounding box (min, max)"""
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
    x: np.ndarray  # X coordinates of vertices
    y: np.ndarray  # Y coordinates of vertices
    z: np.ndarray  # Z coordinates of vertices
    i: List[int]   # Face connectivity indices
    j: List[int]
    k: List[int]
    block_type: str  # medium or conductor
    block_index: int  # original block index
    center: np.ndarray  # center point of the block
    volume: float  # volume of the block
    bounds: Tuple[np.ndarray, np.ndarray]  # bounding box of the block
        
@dataclass
class Layer: 
    """Layer definition with name and type"""
    name: str
    type: str  # interconnect, via, etc.
    
@dataclass
class PolyElement:
    """Polygonal element with custom geometry"""
    name: str
    parent_name: str
    base: np.ndarray
    v1: np.ndarray
    v2: np.ndarray
    hvector: np.ndarray
    coordinates: List[Tuple[float, float]]  # from <coord> tag

    @property
    def bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get bounding box for polygon"""
        # Calculate bounds from base vectors
        base_bounds = np.array([
            self.base,
            self.base + self.v1,
            self.base + self.v2,
            self.base + self.hvector,
            self.base + self.v1 + self.v2,
            self.base + self.v1 + self.hvector,
            self.base + self.v2 + self.hvector,
            self.base + self.v2 + self.v1 + self.hvector
        ])
        
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
    """Simulation window/boundary definition"""
    name: Optional[str]
    v1: np.ndarray  # Corner 1
    v2: np.ndarray  # Corner 2
    boundary_type: Optional[str] = None  # e.g., 'dirichlet'


@dataclass
class Task:
    """Simulation task definition"""
    capacitance_targets: List[str]  # List of conductor names for capacitance calculations


@dataclass
class ParsedCap3DData:
    """Complete parsed CAP3D data structure"""
    blocks: List[Block]
    poly_elements: List[PolyElement]
    layers: List[Layer]
    window: Optional[Window]
    task: Optional[Task]
    stats: Dict 
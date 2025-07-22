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
        
if __name__ == "__main__":
    # Example values for a block
    base = [0, 0, 0]
    v1 = [1, 0, 0]
    v2 = [0, 1, 0]
    hvec = [0, 0, 1]
    block = Block("test_block", "medium", "parent", base, v1, v2, hvec, diel=2.5)
    
    print("Block name:", block.name)
    print("Block vertices:\n", block.vertices)
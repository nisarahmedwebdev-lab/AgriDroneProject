"""
grid_utils.py - Grid loading and utility functions
"""

import json
import numpy as np
from config import GRID_SIZE

def load_field_from_json(file_path):
    """
    Load field configuration from JSON file.
    Returns grid_config, obstacles, disease_seeds, name, crop_type
    """
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    grid_config = data.get('grid_configuration', {})
    obstacles = data.get('obstacles', [])
    disease_seeds = data.get('disease_seeds', [])
    name = data.get('name', 'Unknown Field')
    crop_type = data.get('crop_type', 'Wheat')
    
    # Validate grid dimensions
    rows = grid_config.get('rows', GRID_SIZE)
    cols = grid_config.get('cols', GRID_SIZE)
    
    return {
        'grid_config': grid_config,
        'obstacles': obstacles,
        'disease_seeds': disease_seeds,
        'name': name,
        'crop_type': crop_type,
        'rows': rows,
        'cols': cols
    }

def create_grid_from_json(field_data):
    """
    Create NDVI grid from field configuration.
    """
    rows = field_data.get('rows', GRID_SIZE)
    cols = field_data.get('cols', GRID_SIZE)
    
    # Initialize grid with random NDVI values
    grid = np.random.uniform(0.3, 0.8, size=(rows, cols))
    
    # Mark obstacles as -1 (not scannable)
    obstacles = field_data.get('obstacles', [])
    for obs in obstacles:
        if 0 <= obs[0] < rows and 0 <= obs[1] < cols:
            grid[obs[0], obs[1]] = -1
    
    return grid

def validate_field_data(field_data):
    """
    Validate field data structure.
    """
    required_keys = ['grid_config', 'obstacles', 'disease_seeds', 'name', 'crop_type']
    for key in required_keys:
        if key not in field_data:
            raise ValueError(f"Missing required key: {key}")
    
    # Validate disease seeds
    for seed in field_data.get('disease_seeds', []):
        if 'cell' not in seed or 'type' not in seed:
            raise ValueError("Disease seed must have 'cell' and 'type'")
        if seed['type'] not in ['early', 'severe']:
            raise ValueError("Disease type must be 'early' or 'severe'")
    
    return True

def generate_random_grid(seed=42):
    """
    Generate a random 25x25 grid for testing.
    """
    np.random.seed(seed)
    grid = np.random.uniform(0.3, 0.8, size=(GRID_SIZE, GRID_SIZE))
    
    # Add some obstacles randomly
    obstacles = []
    for _ in range(5):
        r = np.random.randint(0, GRID_SIZE)
        c = np.random.randint(0, GRID_SIZE)
        obstacles.append([r, c])
        grid[r, c] = -1
    
    return grid, obstacles
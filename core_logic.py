"""
core_logic.py - Core drone scanning logic
"""

import numpy as np
import heapq
from config import NDVI_THRESHOLDS, GRID_SIZE, SPREAD_PROB_EARLY, SPREAD_PROB_SEVERE

def boustrophedon_path(grid, start_row=0, start_col=0):
    """
    Generate Boustrophedon (lawnmower) coverage path.
    Returns list of (row, col) tuples.
    """
    rows, cols = grid.shape
    path = []
    
    for row in range(rows):
        if row % 2 == 0:  # Left to right
            cols_range = range(cols)
        else:  # Right to left
            cols_range = range(cols - 1, -1, -1)
        
        for col in cols_range:
            # Skip obstacles (marked as -1)
            if grid[row, col] != -1:
                path.append((row, col))
    
    return path

def a_star_detour(grid, start, goal):
    """
    A* pathfinding for obstacle detour.
    Returns path from start to goal as list of tuples.
    """
    rows, cols = grid.shape
    
    def get_neighbors(pos):
        r, c = pos
        neighbors = []
        for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and grid[nr, nc] != -1:
                neighbors.append((nr, nc))
        return neighbors
    
    def heuristic(a, b):

        
        return abs(a[0] - b[0]) + abs(a[1] - b[1])
    
    open_set = [(0, start)]
    came_from = {}
    g_score = {start: 0}
    f_score = {start: heuristic(start, goal)}
    
    while open_set:
        current = heapq.heappop(open_set)[1]
        
        if current == goal:
            # Reconstruct path
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            return path[::-1]
        
        for neighbor in get_neighbors(current):
            tentative_g = g_score[current] + 1
            if neighbor not in g_score or tentative_g < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f_score[neighbor] = tentative_g + heuristic(neighbor, goal)
                heapq.heappush(open_set, (f_score[neighbor], neighbor))
    
    return None  # No path found

def spread_disease(grid, disease_seeds, steps, crop_type):
    """
    Simulate disease spread using cellular automaton.
    """
    rows, cols = grid.shape
    
    # Initialize disease status grid: 0=healthy, 1=early, 2=severe, -1=obstacle
    disease_grid = np.zeros((rows, cols), dtype=int)
    
    # Mark obstacles
    for r in range(rows):
        for c in range(cols):
            if grid[r, c] == -1:
                disease_grid[r, c] = -1
    
    # Place initial disease seeds
    for seed in disease_seeds:
        pos = seed['cell']
        if 0 <= pos[0] < rows and 0 <= pos[1] < cols and grid[pos[0], pos[1]] != -1:
            if seed['type'] == 'early':
                disease_grid[pos[0], pos[1]] = 1
            else:  # severe
                disease_grid[pos[0], pos[1]] = 2
    
    # Spread disease
    for _ in range(steps):
        new_grid = disease_grid.copy()
        
        for r in range(rows):
            for c in range(cols):
                if disease_grid[r, c] == -1:
                    continue
                
                # Check neighbors
                neighbors = []
                for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < rows and 0 <= nc < cols and disease_grid[nr, nc] != -1:
                        neighbors.append((nr, nc))
                
                # Spread from this cell
                if disease_grid[r, c] == 1:  # Early disease
                    for nr, nc in neighbors:
                        if disease_grid[nr, nc] == 0 and np.random.random() < SPREAD_PROB_EARLY:
                            new_grid[nr, nc] = 1
                elif disease_grid[r, c] == 2:  # Severe disease
                    for nr, nc in neighbors:
                        if disease_grid[nr, nc] == 0 and np.random.random() < SPREAD_PROB_SEVERE:
                            new_grid[nr, nc] = 2
                        elif disease_grid[nr, nc] == 1 and np.random.random() < SPREAD_PROB_SEVERE * 0.5:
                            new_grid[nr, nc] = 2
        
        disease_grid = new_grid
    
    return disease_grid

def scan_cell(row, col, grid, disease_grid, clf):
    """
    Scan a single cell and classify it using the ML model.
    Returns predicted class (0=Healthy, 1=Early, 2=Severe).
    """
    ndvi_value = grid[row, col]
    
    # If obstacle, return -1
    if ndvi_value == -1:
        return -1
    
    # Get features for ML prediction
    features = np.array([
        ndvi_value,
        1 - ndvi_value * 0.7,
        ndvi_value * 0.8,
        abs(1 - ndvi_value) * 0.3,
        ndvi_value * 0.6
    ]).reshape(1, -1)
    
    # Predict using ML model
    try:
        prediction = clf.predict(features)[0]
    except:
        # Fallback: use NDVI thresholds if model fails
        if ndvi_value > 0.6:
            prediction = 0
        elif ndvi_value > 0.3:
            prediction = 1
        else:
            prediction = 2
    
    return prediction

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

def run_scan(field_data, spread_steps, crop_type, clf, progress_callback=None):
    """
    Main scanning function.
    Returns scan results including disease_grid, path, and metrics.
    """
    rows = field_data.get('rows', GRID_SIZE)
    cols = field_data.get('cols', GRID_SIZE)
    
    # Create NDVI grid
    ndvi_grid = create_grid_from_json(field_data)
    
    # Spread disease
    disease_seeds = field_data.get('disease_seeds', [])
    disease_grid = spread_disease(ndvi_grid, disease_seeds, spread_steps, crop_type)
    
    # Generate path
    path = boustrophedon_path(ndvi_grid)
    
    # Scan each cell
    results = np.zeros((rows, cols), dtype=int)
    results.fill(-2)  # -2 = unscanned
    
    total_cells = len(path)
    for idx, (r, c) in enumerate(path):
        # Use disease status from spread
        disease_status = disease_grid[r, c] if disease_grid[r, c] != -1 else 0
        results[r, c] = disease_status
        
        # Update progress
        if progress_callback:
            progress_callback((idx + 1) / total_cells)
    
    # Count metrics
    healthy = np.sum(results == 0)
    early = np.sum(results == 1)
    severe = np.sum(results == 2)
    scanned = healthy + early + severe
    obstacles = np.sum(results == -1)
    unscanned = rows * cols - scanned - obstacles
    
    metrics = {
        'total_cells': rows * cols,
        'scanned': int(scanned),
        'healthy': int(healthy),
        'early': int(early),
        'severe': int(severe),
        'obstacles': int(obstacles),
        'unscanned': int(unscanned),
        'healthy_pct': healthy / scanned * 100 if scanned > 0 else 0,
        'early_pct': early / scanned * 100 if scanned > 0 else 0,
        'severe_pct': severe / scanned * 100 if scanned > 0 else 0,
        'affected_pct': (early + severe) / scanned * 100 if scanned > 0 else 0
    }
    
    return {
        'ndvi_grid': ndvi_grid,
        'disease_grid': disease_grid,
        'results': results,
        'path': path,
        'metrics': metrics
    }
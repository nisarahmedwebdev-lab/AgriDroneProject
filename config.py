"""
config.py - Configuration constants for AgriDrone
"""

# Disease class labels
DISEASE_CLASSES = {
    0: "Healthy",
    1: "Early Disease",
    2: "Severe Disease"
}

# Color mapping for disease classes (for heatmap)
CLASS_COLORS = {
    "Healthy": "#2ecc71",      
    "Early Disease": "#f1c40f", 
    "Severe Disease": "#e74c3c", 
    "Obstacle": "#34495e",      
    "Unscanned": "#ecf0f1"      
}

# NDVI thresholds per crop type
NDVI_THRESHOLDS = {
    "Wheat": {
        "healthy_min": 0.6,
        "early_min": 0.3,
        "severe_min": 0.0
    },
    "Cotton": {
        "healthy_min": 0.55,
        "early_min": 0.25,
        "severe_min": 0.0
    },
    "Rice": {
        "healthy_min": 0.5,
        "early_min": 0.2,
        "severe_min": 0.0
    },
    "Sugarcane": {
        "healthy_min": 0.65,
        "early_min": 0.35,
        "severe_min": 0.0
    }
}

# Default NDVI range for synthetic data generation
NDVI_RANGE = (0.0, 1.0)

# Grid size
GRID_SIZE = 25

# Disease spread probabilities
SPREAD_PROB_EARLY = 0.3
SPREAD_PROB_SEVERE = 0.6

# Gemini model configuration
GEMINI_MODEL = "gemini-1.5-flash"
GEMINI_TEMPERATURE = 0.5

# Feature names for ML model
FEATURE_NAMES = [
    "NDVI",
    "red_intensity",
    "green_intensity",
    "texture_variance",
    "moisture_index"
]
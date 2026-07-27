"""
llm_handler.py - API handler for AI report generation
Supports both Groq and OpenAI APIs
"""

import os
import requests
from typing import Dict, List, Optional
from dotenv import load_dotenv

load_dotenv()

# Try Groq first, fallback to OpenAI
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Use Groq by default if available
MODEL_NAME = "llama-3.3-70b-versatile"  # Groq model

def initialize_openai():
    """Check if API key is configured (Groq or OpenAI)."""
    if GROQ_API_KEY and GROQ_API_KEY.startswith("gsk_"):
        return True
    if OPENAI_API_KEY and OPENAI_API_KEY.startswith("sk-"):
        return True
    return False

def init_gemini():
    """Alias for initialize_openai for backward compatibility."""
    return initialize_openai()

def generate_report(metrics: Dict, crop_type: str, field_name: str, disease_seeds: Optional[List] = None) -> str:
    """Generate agronomist report using available API.
    
    Args:
        metrics: Dictionary with scan metrics
        crop_type: Type of crop (Wheat, Cotton, etc.)
        field_name: Name of the field
        disease_seeds: List of disease seed points (optional)
    """
    
    # Try Groq first
    if GROQ_API_KEY and GROQ_API_KEY.startswith("gsk_"):
        return generate_report_groq(metrics, crop_type, field_name, disease_seeds)
    
    # Fallback to OpenAI
    if OPENAI_API_KEY and OPENAI_API_KEY.startswith("sk-"):
        return generate_report_openai(metrics, crop_type, field_name, disease_seeds)
    
    return "⚠️ No API key found. Please set GROQ_API_KEY or OPENAI_API_KEY in .env file."

def get_spray_advice(metrics: Dict, crop_type: str, disease_seeds: Optional[List] = None) -> str:
    """Get spray advice using available API.
    
    Args:
        metrics: Dictionary with scan metrics
        crop_type: Type of crop (Wheat, Cotton, etc.)
        disease_seeds: List of disease seed points (optional)
    """
    
    if GROQ_API_KEY and GROQ_API_KEY.startswith("gsk_"):
        return get_spray_advice_groq(metrics, crop_type, disease_seeds)
    
    if OPENAI_API_KEY and OPENAI_API_KEY.startswith("sk-"):
        return get_spray_advice_openai(metrics, crop_type, disease_seeds)
    
    return "⚠️ No API key found. Please set GROQ_API_KEY or OPENAI_API_KEY in .env file."

# ── Groq Functions ──────────────────────────────────────────────

def generate_report_groq(metrics: Dict, crop_type: str, field_name: str, disease_seeds: Optional[List] = None) -> str:
    """Generate report using Groq API."""
    healthy_count = metrics.get('healthy', 0)
    early_count = metrics.get('early', 0)
    severe_count = metrics.get('severe', 0)
    total = metrics.get('total_cells', 1)
    scanned = metrics.get('scanned', 0)
    
    healthy_pct = (healthy_count / total) * 100 if total > 0 else 0
    early_pct = (early_count / total) * 100 if total > 0 else 0
    severe_pct = (severe_count / total) * 100 if total > 0 else 0
    
    # Add seed information if provided
    seed_info = ""
    if disease_seeds:
        seed_info = f"\nDisease Seeds: {len(disease_seeds)} initial infection points"
        for i, seed in enumerate(disease_seeds[:3]):  # Show first 3 seeds
            if isinstance(seed, dict):
                cell = seed.get('cell', [0, 0])
                seed_type = seed.get('type', 'unknown')
                seed_info += f"\n  - Seed {i+1}: Position ({cell[0]}, {cell[1]}), Type: {seed_type}"
        if len(disease_seeds) > 3:
            seed_info += f"\n  - ... and {len(disease_seeds) - 3} more seeds"
    
    prompt = f"""
You are an agricultural AI assistant. Generate a concise field health report.

Field: {field_name} | Crop: {crop_type}
Scanned: {scanned} cells | Total: {total} cells
Healthy: {healthy_count} ({healthy_pct:.1f}%)
Early disease: {early_count} ({early_pct:.1f}%)
Severe disease: {severe_count} ({severe_pct:.1f}%){seed_info}

Write 4 short paragraphs on:
1. Overall field health summary
2. Disease risk assessment
3. Treatment recommendations
4. Monitoring schedule

Use plain language. No markdown. Keep it concise.
"""
    try:
        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json"
        }
        data = {
            "model": MODEL_NAME,
            "messages": [
                {"role": "system", "content": "You are an agricultural AI assistant."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.5,
            "max_tokens": 500
        }
        response = requests.post(url, headers=headers, json=data, timeout=30)
        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"]
        else:
            return f"⚠️ Groq Error: {response.text[:200]}"
    except Exception as e:
        return f"⚠️ Error: {str(e)}"

def get_spray_advice_groq(metrics: Dict, crop_type: str, disease_seeds: Optional[List] = None) -> str:
    """Get spray advice using Groq API."""
    affected = metrics.get('early', 0) + metrics.get('severe', 0)
    total = metrics.get('total_cells', 1)
    scanned = metrics.get('scanned', 0)
    affected_pct = (affected / total) * 100 if total > 0 else 0
    
    # Add seed information if provided
    seed_info = ""
    if disease_seeds:
        early_seeds = sum(1 for s in disease_seeds if isinstance(s, dict) and s.get('type') == 'early')
        severe_seeds = sum(1 for s in disease_seeds if isinstance(s, dict) and s.get('type') == 'severe')
        if early_seeds or severe_seeds:
            seed_info = f"\nInitial infection sources: {early_seeds} early, {severe_seeds} severe"
    
    prompt = f"""
Crop: {crop_type}
Scanned: {scanned} cells
Affected: {affected_pct:.1f}% of the field
Early disease: {metrics.get('early', 0)} cells
Severe disease: {metrics.get('severe', 0)} cells{seed_info}

Provide concise spray recommendation including:
1. Recommended fungicide/pesticide type
2. Application method and timing
3. Dosage guidance
4. Safety precautions

2-3 short paragraphs. Plain language.
"""
    try:
        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json"
        }
        data = {
            "model": MODEL_NAME,
            "messages": [
                {"role": "system", "content": "You are a crop protection expert."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.5,
            "max_tokens": 400
        }
        response = requests.post(url, headers=headers, json=data, timeout=30)
        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"]
        else:
            return f"⚠️ Groq Error: {response.text[:200]}"
    except Exception as e:
        return f"⚠️ Error: {str(e)}"

# ── OpenAI Functions (Fallback) ─────────────────────────────────

def generate_report_openai(metrics: Dict, crop_type: str, field_name: str, disease_seeds: Optional[List] = None) -> str:
    """Generate report using OpenAI API (fallback)."""
    if not OPENAI_API_KEY:
        return "⚠️ OpenAI API key not configured."
    
    healthy_count = metrics.get('healthy', 0)
    early_count = metrics.get('early', 0)
    severe_count = metrics.get('severe', 0)
    total = metrics.get('total_cells', 1)
    scanned = metrics.get('scanned', 0)
    
    healthy_pct = (healthy_count / total) * 100 if total > 0 else 0
    early_pct = (early_count / total) * 100 if total > 0 else 0
    severe_pct = (severe_count / total) * 100 if total > 0 else 0
    
    prompt = f"""
You are an agricultural AI assistant. Generate a concise field health report.

Field: {field_name} | Crop: {crop_type}
Scanned: {scanned} cells
Healthy: {healthy_count} ({healthy_pct:.1f}%)
Early disease: {early_count} ({early_pct:.1f}%)
Severe disease: {severe_count} ({severe_pct:.1f}%)

Write 4 short paragraphs on: overall summary, disease risk, treatments, and monitoring schedule.
"""
    try:
        url = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "Content-Type": "application/json"
        }
        data = {
            "model": "gpt-3.5-turbo",
            "messages": [
                {"role": "system", "content": "You are an agricultural AI assistant."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.5,
            "max_tokens": 500
        }
        response = requests.post(url, headers=headers, json=data, timeout=30)
        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"]
        else:
            return f"⚠️ OpenAI Error: {response.text[:200]}"
    except Exception as e:
        return f"⚠️ Error: {str(e)}"

def get_spray_advice_openai(metrics: Dict, crop_type: str, disease_seeds: Optional[List] = None) -> str:
    """Get spray advice using OpenAI API (fallback)."""
    if not OPENAI_API_KEY:
        return "⚠️ OpenAI API key not configured."
    
    affected = metrics.get('early', 0) + metrics.get('severe', 0)
    total = metrics.get('total_cells', 1)
    affected_pct = (affected / total) * 100 if total > 0 else 0
    
    prompt = f"""
Crop: {crop_type}
Affected: {affected_pct:.1f}% of the field
Early disease: {metrics.get('early', 0)} cells
Severe disease: {metrics.get('severe', 0)} cells

Provide concise spray recommendation: type, method, dosage, safety. 2-3 paragraphs.
"""
    try:
        url = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "Content-Type": "application/json"
        }
        data = {
            "model": "gpt-3.5-turbo",
            "messages": [
                {"role": "system", "content": "You are a crop protection expert."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.5,
            "max_tokens": 400
        }
        response = requests.post(url, headers=headers, json=data, timeout=30)
        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"]
        else:
            return f"⚠️ OpenAI Error: {response.text[:200]}"
    except Exception as e:
        return f"⚠️ Error: {str(e)}"
"""
llm_handler.py - Gemini API integration for agronomist reports
"""

import os
import google.generativeai as genai
from config import GEMINI_MODEL, GEMINI_TEMPERATURE

def init_gemini(api_key=None):
    """
    Initialize Gemini API with provided key or from environment.
    """
    if api_key is None:
        api_key = os.getenv('GEMINI_API_KEY')
    
    if not api_key:
        return False
    
    try:
        genai.configure(api_key=api_key)
        return True
    except Exception as e:
        print(f"Error initializing Gemini: {e}")
        return False

def generate_report(metrics, crop_type, field_name="Unknown Field"):
    """
    Generate an agronomist report using Gemini.
    """
    try:
        # Check if Gemini is configured
        try:
            genai.get_model(f"models/{GEMINI_MODEL}")
        except:
            return "Gemini API not configured. Please set GEMINI_API_KEY in .env file."
        
        prompt = f"""
You are an agricultural AI assistant.
Generate a concise field health report for a farmer.

Field: {field_name}
Crop: {crop_type}
Grid: 25x25

Scan Results:
  Healthy cells  : {metrics['healthy']} ({metrics['healthy_pct']:.1f}%)
  Early disease  : {metrics['early']}   ({metrics['early_pct']:.1f}%)
  Severe disease : {metrics['severe']}  ({metrics['severe_pct']:.1f}%)
  Affected area  : {metrics['affected_pct']:.1f}%

Write exactly 4 paragraphs:
  1. Overall field health summary
  2. Disease risk and likely spread pattern
  3. Specific treatment recommendations
  4. Next monitoring schedule

Plain language. No bullet points. No markdown.
"""
        
        model = genai.GenerativeModel(
            GEMINI_MODEL,
            generation_config=genai.types.GenerationConfig(
                temperature=GEMINI_TEMPERATURE
            )
        )
        
        response = model.generate_content(prompt)
        return response.text
        
    except Exception as e:
        return f"Error generating report: {str(e)}"

def get_spray_advice(metrics, crop_type, disease_seeds):
    """
    Generate targeted spray advice using Gemini.
    """
    try:
        # Check if Gemini is configured
        try:
            genai.get_model(f"models/{GEMINI_MODEL}")
        except:
            return "Gemini API not configured. Please set GEMINI_API_KEY in .env file."
        
        prompt = f"""
You are an agricultural AI assistant.
Provide specific pesticide/fungicide spray advice for a farmer.

Crop: {crop_type}
Disease severity: {metrics['severe']} cells severe ({metrics['severe_pct']:.1f}%)
Early disease: {metrics['early']} cells ({metrics['early_pct']:.1f}%)
Number of disease seeds: {len(disease_seeds)}

Provide recommendations:
1. Type of fungicide/pesticide needed
2. Application rate (per acre)
3. Best time to spray
4. Safety precautions
5. Expected results

Be specific and practical. Keep it to 3-4 paragraphs.
"""
        
        model = genai.GenerativeModel(
            GEMINI_MODEL,
            generation_config=genai.types.GenerationConfig(
                temperature=GEMINI_TEMPERATURE
            )
        )
        
        response = model.generate_content(prompt)
        return response.text
        
    except Exception as e:
        return f"Error generating spray advice: {str(e)}"
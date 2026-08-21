# llm_handler.py - OpenAI Version
import os
from dotenv import load_dotenv
from openai import OpenAI
from datetime import datetime

load_dotenv()

# Initialize OpenAI client
client = None

def init_openai(api_key=None):
    """Initialize OpenAI API"""
    global client
    
    if api_key is None:
        api_key = os.getenv('OPENAI_API_KEY')
    
    if not api_key:
        return False
    
    try:
        client = OpenAI(api_key=api_key)
        # Test the connection
        test_response = client.chat.completions.create(
            messages=[{"role": "user", "content": "test"}],
            model="gpt-3.5-turbo",
            max_tokens=5,
        )
        return True
    except Exception as e:
        print(f"Error initializing OpenAI: {e}")
        return False

def generate_report(metrics, crop_type, field_name="Unknown Field", disease_seeds=None):
    """Generate report using OpenAI"""
    global client
    
    try:
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            return get_fallback_report(metrics, crop_type, field_name)
        
        if client is None:
            client = OpenAI(api_key=api_key)
        
        prompt = f"""
You are an agricultural AI assistant. Generate a concise field health report.

Field: {field_name}
Crop: {crop_type}

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

Use plain language. No bullet points. No markdown.
"""
        
        response = client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are an agricultural AI assistant."},
                {"role": "user", "content": prompt}
            ],
            model="gpt-3.5-turbo",
            temperature=0.5,
            max_tokens=600,
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        return f"⚠️ OpenAI Error: {str(e)}\n\n{get_fallback_report(metrics, crop_type, field_name)}"

def get_spray_advice(metrics, crop_type, disease_seeds=None):
    """Generate spray advice using OpenAI"""
    global client
    
    try:
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            return get_fallback_spray_advice(metrics, crop_type)
        
        if client is None:
            client = OpenAI(api_key=api_key)
        
        prompt = f"""
You are an agricultural AI assistant. Provide specific spray advice.

Crop: {crop_type}
Severe disease: {metrics['severe']} cells ({metrics['severe_pct']:.1f}%)
Early disease: {metrics['early']} cells ({metrics['early_pct']:.1f}%)
Affected area: {metrics['affected_pct']:.1f}%

Provide recommendations:
1. Type of fungicide/pesticide needed
2. Application rate per acre
3. Best time to spray
4. Safety precautions

Be specific and practical.
"""
        
        response = client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are an agricultural AI assistant."},
                {"role": "user", "content": prompt}
            ],
            model="gpt-3.5-turbo",
            temperature=0.5,
            max_tokens=600,
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        return f"⚠️ OpenAI Error: {str(e)}\n\n{get_fallback_spray_advice(metrics, crop_type)}"

def get_fallback_report(metrics, crop_type, field_name):
    """Generate fallback report when API is not available"""
    severity = "High" if metrics['severe_pct'] > 15 else "Medium" if metrics['severe_pct'] > 5 else "Low"
    
    return f"""
================================================================================
                              FIELD HEALTH REPORT
================================================================================

Field Name: {field_name}
Crop Type: {crop_type}
Report Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Severity Level: {severity}

================================================================================
                              SCAN RESULTS
================================================================================

Total Cells Scanned: {metrics['scanned']}/{metrics['total_cells']}

Status Distribution:
  ✅ Healthy        : {metrics['healthy']} ({metrics['healthy_pct']:.1f}%)
  🟡 Early Disease  : {metrics['early']} ({metrics['early_pct']:.1f}%)
  🔴 Severe Disease : {metrics['severe']} ({metrics['severe_pct']:.1f}%)

================================================================================
                              RECOMMENDATIONS
================================================================================

1. IMMEDIATE ACTIONS:
   {'🔴 Apply fungicide to severe disease areas immediately' if metrics['severe'] > 50 else '🟡 Monitor severe disease areas daily'}
   {'🟡 Apply preventive treatment to early disease areas' if metrics['early'] > 50 else '🟢 Continue regular monitoring'}

2. SHORT-TERM (Next 3-5 days):
   • {'Apply fungicide to early disease areas' if metrics['early'] > 30 else 'Monitor early disease areas'}
   • Check moisture levels in {'severe disease' if metrics['severe'] > metrics['early'] else 'early disease'} areas
   • {'Quarantine affected areas' if metrics['affected_pct'] > 20 else 'Mark affected zones for treatment'}

3. LONG-TERM (Next 2 weeks):
   • Schedule next drone scan in 5-7 days
   • {'Consider crop rotation for next season' if metrics['affected_pct'] > 30 else 'Continue regular monitoring'}
   • Keep records of all treatments applied

================================================================================
"""

def get_fallback_spray_advice(metrics, crop_type):
    """Generate fallback spray advice"""
    return f"""
================================================================================
                           SPRAY ADVICE RECOMMENDATIONS
================================================================================

Crop: {crop_type}
Severity Level: {'High' if metrics['severe_pct'] > 15 else 'Medium' if metrics['severe_pct'] > 5 else 'Low'}

================================================================================
                           RECOMMENDED SPRAY SCHEDULE
================================================================================

1. FUNGICIDE/PESTICIDE TYPE:
   {'Broad-spectrum fungicide' if metrics['severe_pct'] > 10 else 'Preventive fungicide'}

2. APPLICATION RATE:
   {'2-3 liters per acre (High concentration)' if metrics['severe_pct'] > 10 else '1-2 liters per acre (Standard concentration)'}

3. BEST TIME TO SPRAY:
   • Early morning (6:00 AM - 9:00 AM)
   • Late afternoon (4:00 PM - 6:00 PM)
   • Avoid spraying during windy conditions
   • Avoid spraying before rain

4. SAFETY PRECAUTIONS:
   • Wear protective equipment (gloves, mask, goggles)
   • Keep children and animals away during application
   • Follow manufacturer's instructions
   • Dispose of containers properly

5. FOLLOW-UP:
   • Check effectiveness after 3-5 days
   • Apply second spray if needed after 7-10 days
   • Monitor for any adverse effects

================================================================================
"""
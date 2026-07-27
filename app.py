"""
app.py - Main Streamlit Application for AgriDrone
"""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import pandas as pd
import json
import os
from pathlib import Path
import joblib

# Import modules
from config import GRID_SIZE
from core_logic import run_scan
from llm_handler import initialize_openai, generate_report, get_spray_advice

# Page configuration
st.set_page_config(
    page_title="AgriDrone - Crop Disease Monitor",
    layout="wide",
    page_icon="🚁"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #2e7d32;
    }
    .stProgress > div > div {
        background-color: #2e7d32;
    }
    .report-text {
        font-family: 'Courier New', monospace;
        background-color: #f8f9fa;
        padding: 20px;
        border-radius: 10px;
        border-left: 4px solid #2e7d32;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'scan_results' not in st.session_state:
    st.session_state.scan_results = None
if 'field_data' not in st.session_state:
    st.session_state.field_data = None
if 'metrics' not in st.session_state:
    st.session_state.metrics = None
if 'scan_complete' not in st.session_state:
    st.session_state.scan_complete = False
if 'field_name' not in st.session_state:
    st.session_state.field_name = "No field loaded"
if 'crop_type' not in st.session_state:
    st.session_state.crop_type = "Wheat"
if 'report_text' not in st.session_state:
    st.session_state.report_text = ""

@st.cache_resource
def load_model():
    """Load the trained Random Forest model."""
    model_path = Path('models/disease_clf.pkl')
    if model_path.exists():
        return joblib.load(model_path)
    else:
        st.error("Model not found. Please run disease_model.py first.")
        return None

def create_default_fields():
    """Create default field configuration files."""
    data_dir = Path('data')
    data_dir.mkdir(exist_ok=True)
    
    # flat_farm.json
    flat_farm = {
        "name": "Flat Farm Layout",
        "crop_type": "Wheat",
        "grid_configuration": {
            "rows": GRID_SIZE,
            "cols": GRID_SIZE,
            "drone_start": [0, 0]
        },
        "obstacles": [],
        "disease_seeds": [
            {"cell": [5, 5], "type": "early"},
            {"cell": [12, 18], "type": "early"},
            {"cell": [20, 7], "type": "severe"},
            {"cell": [8, 20], "type": "early"},
            {"cell": [15, 15], "type": "severe"}
        ]
    }
    
    # pond_farm.json
    pond_farm = {
        "name": "Pond Farm Layout",
        "crop_type": "Cotton",
        "grid_configuration": {
            "rows": GRID_SIZE,
            "cols": GRID_SIZE,
            "drone_start": [0, 0]
        },
        "obstacles": [[10, 10], [10, 11], [11, 10], [11, 11]],
        "disease_seeds": [
            {"cell": [5, 5], "type": "early"},
            {"cell": [18, 3], "type": "severe"},
            {"cell": [3, 18], "type": "early"}
        ]
    }
    
    # dense_field.json
    dense_field = {
        "name": "Dense Field Layout",
        "crop_type": "Rice",
        "grid_configuration": {
            "rows": GRID_SIZE,
            "cols": GRID_SIZE,
            "drone_start": [0, 0]
        },
        "obstacles": [[7, 7], [7, 8], [8, 7]],
        "disease_seeds": [
            {"cell": [3, 3], "type": "early"},
            {"cell": [20, 20], "type": "severe"},
            {"cell": [12, 12], "type": "early"},
            {"cell": [5, 20], "type": "severe"},
            {"cell": [20, 5], "type": "early"}
        ]
    }
    
    # Save files
    with open(data_dir / 'flat_farm.json', 'w') as f:
        json.dump(flat_farm, f, indent=2)
    with open(data_dir / 'pond_farm.json', 'w') as f:
        json.dump(pond_farm, f, indent=2)
    with open(data_dir / 'dense_field.json', 'w') as f:
        json.dump(dense_field, f, indent=2)
    
    return ["flat_farm", "pond_farm", "dense_field"]

def load_field_options():
    """Load available field configurations."""
    data_dir = Path('data')
    
    if not data_dir.exists():
        data_dir.mkdir(exist_ok=True)
        return create_default_fields()
    
    field_files = list(data_dir.glob('*.json'))
    if not field_files:
        return create_default_fields()
    
    return [f.stem for f in field_files]

def create_heatmap(results):
    """Create an interactive Plotly heatmap."""
    rows, cols = results.shape
    
    # Create a numeric mapping for the heatmap
    # 0=Healthy, 1=Early Disease, 2=Severe Disease, -1=Obstacle, -2=Unscanned
    class_names = {0: "Healthy", 1: "Early Disease", 2: "Severe Disease", -1: "Obstacle", -2: "Unscanned"}
    
    # Create custom color scale
    color_map = {
        0: "#2ecc71",      # Healthy - Green
        1: "#f1c40f",      # Early Disease - Yellow
        2: "#e74c3c",      # Severe Disease - Red
        -1: "#34495e",     # Obstacle - Dark Gray
        -2: "#ecf0f1"      # Unscanned - Light Gray
    }
    
    # Convert to numeric for heatmap
    numeric_data = results.copy()
    
    # Create hover text
    hover_text = np.empty((rows, cols), dtype=object)
    for r in range(rows):
        for c in range(cols):
            val = results[r, c]
            hover_text[r, c] = f"Row: {r}<br>Col: {c}<br>Status: {class_names.get(val, 'Unknown')}"
    
    # Create heatmap using go.Heatmap
    fig = go.Figure(data=go.Heatmap(
        z=numeric_data,
        colorscale=[
            [0, color_map[-2]],  # Unscanned
            [0.2, color_map[-1]], # Obstacle
            [0.4, color_map[0]],  # Healthy
            [0.7, color_map[1]],  # Early Disease
            [1.0, color_map[2]]   # Severe Disease
        ],
        text=hover_text,
        hoverinfo='text',
        showscale=False,
        zmin=-2,
        zmax=2
    ))
    
    fig.update_layout(
        title="Field Health Map",
        height=500,
        xaxis=dict(
            title="Column",
            showgrid=False,
            tickmode='linear',
            dtick=1
        ),
        yaxis=dict(
            title="Row",
            showgrid=False,
            autorange='reversed',
            tickmode='linear',
            dtick=1
        ),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=50, r=50, t=50, b=50)
    )
    
    return fig

def create_pie_chart(metrics):
    """Create pie chart for disease distribution."""
    labels = ['Healthy', 'Early Disease', 'Severe Disease']
    values = [metrics['healthy'], metrics['early'], metrics['severe']]
    colors = ['#2ecc71', '#f1c40f', '#e74c3c']
    
    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        marker=dict(colors=colors),
        textinfo='label+percent',
        textposition='inside',
        hole=0.3
    )])
    
    fig.update_layout(
        title="Disease Distribution",
        height=350,
        margin=dict(l=20, r=20, t=40, b=20),
        showlegend=False
    )
    
    return fig

def create_bar_chart(metrics):
    """Create bar chart comparing healthy vs affected."""
    categories = ['Healthy', 'Early Disease', 'Severe Disease']
    values = [metrics['healthy'], metrics['early'], metrics['severe']]
    colors = ['#2ecc71', '#f1c40f', '#e74c3c']
    
    fig = go.Figure(data=[go.Bar(
        x=categories,
        y=values,
        marker_color=colors,
        text=values,
        textposition='auto'
    )])
    
    fig.update_layout(
        title="Disease Counts",
        xaxis_title="Category",
        yaxis_title="Number of Cells",
        height=350,
        margin=dict(l=20, r=20, t=40, b=40),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig

def display_metrics(metrics):
    """Display metric cards."""
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="📊 Cells Scanned",
            value=metrics['scanned'],
            delta=f"{metrics['scanned']}/{metrics['total_cells']} cells"
        )
    
    with col2:
        st.metric(
            label="🟢 Healthy",
            value=metrics['healthy'],
            delta=f"{metrics['healthy_pct']:.1f}%"
        )
    
    with col3:
        st.metric(
            label="🟡 Early Disease",
            value=metrics['early'],
            delta=f"{metrics['early_pct']:.1f}%",
            delta_color="inverse"
        )
    
    with col4:
        st.metric(
            label="🔴 Severe Disease",
            value=metrics['severe'],
            delta=f"{metrics['severe_pct']:.1f}%",
            delta_color="inverse"
        )

def main():
    """Main app function."""
    # Load model
    clf = load_model()
    
    # Title in main area
    st.markdown('<p class="main-header">🚁 Agricultural Drone - Crop Disease Monitor</p>', unsafe_allow_html=True)
    
    # SIDEBAR
    with st.sidebar:
        st.markdown("### 🌱 AgriDrone Controls")
        st.markdown("---")
        
        # Crop selection
        crop_type = st.selectbox(
            "🌾 Crop Type",
            ["Wheat", "Cotton", "Rice", "Sugarcane"],
            index=0,
            help="Select the crop type for NDVI threshold adjustment"
        )
        st.session_state.crop_type = crop_type
        
        # Field selection
        field_options = load_field_options()
        field_choice = st.selectbox(
            "🗺️ Field Layout",
            field_options,
            help="Select a pre-configured field layout"
        )
        
        # File uploader
        uploaded_file = st.file_uploader(
            "📤 Upload Custom Field",
            type=["json"],
            help="Upload a JSON file with custom field configuration"
        )
        
        if uploaded_file is not None:
            try:
                field_data = json.load(uploaded_file)
                # Ensure rows and cols are set
                if 'grid_configuration' in field_data:
                    field_data['rows'] = field_data['grid_configuration'].get('rows', GRID_SIZE)
                    field_data['cols'] = field_data['grid_configuration'].get('cols', GRID_SIZE)
                else:
                    field_data['rows'] = GRID_SIZE
                    field_data['cols'] = GRID_SIZE
                st.session_state.field_data = field_data
                st.session_state.field_name = field_data.get('name', 'Custom Field')
                st.session_state.crop_type = field_data.get('crop_type', crop_type)
                st.success(f"✅ Loaded: {st.session_state.field_name}")
            except Exception as e:
                st.error(f"❌ Error loading file: {e}")
        elif field_choice and field_choice not in ["No fields found. Please upload a JSON file."]:
            # Load selected field
            try:
                field_path = Path('data') / f'{field_choice}.json'
                if field_path.exists():
                    with open(field_path, 'r') as f:
                        field_data = json.load(f)
                    # Ensure rows and cols are set
                    if 'grid_configuration' in field_data:
                        field_data['rows'] = field_data['grid_configuration'].get('rows', GRID_SIZE)
                        field_data['cols'] = field_data['grid_configuration'].get('cols', GRID_SIZE)
                    else:
                        field_data['rows'] = GRID_SIZE
                        field_data['cols'] = GRID_SIZE
                    st.session_state.field_data = field_data
                    st.session_state.field_name = field_data.get('name', field_choice)
                    st.session_state.crop_type = field_data.get('crop_type', crop_type)
            except Exception as e:
                st.error(f"❌ Error loading field: {e}")
        
        st.markdown("---")
        
        # Display current field info
        if st.session_state.field_data:
            st.caption(f"📍 Current Field: {st.session_state.field_name}")
            st.caption(f"🌾 Crop: {st.session_state.crop_type}")
        
        st.markdown("---")
        
        # Disease spread steps
        spread_steps = st.slider(
            "🔄 Disease Spread Steps",
            min_value=0,
            max_value=10,
            value=5,
            help="Number of disease spread iterations before scanning"
        )
        
        st.markdown("---")
        
        # Run button
        run_clicked = st.button(
            "▶️ Run Drone Scan",
            type="primary",
            width='stretch',
            disabled=clf is None
        )
        
        st.markdown("---")
        
        # API status
        api_ok = initialize_openai()
        if api_ok:
            st.success("✅ Groq/OpenAI: Connected")
        else:
            st.warning("⚠️ API: Not configured")
            st.caption("Set GROQ_API_KEY or OPENAI_API_KEY in .env file")
    
    # MAIN AREA
    
    # Check if field is loaded
    if st.session_state.field_data is None:
        st.info("📋 Load a field from the sidebar and click '▶️ Run Drone Scan' to start.")
        
        with st.expander("📖 How it works"):
            st.markdown("""
            **🚁 AgriDrone simulates an agricultural drone scanning a 25x25 farm grid:**
            
            1. **🗺️ Boustrophedon Path** - Drone follows a lawnmower pattern
            2. **🔬 Disease Detection** - Random Forest classifier analyzes NDVI values
            3. **📊 Visualization** - Interactive heatmap shows disease status
            4. **🤖 AI Reports** - Gemini/Groq generates agronomist recommendations
            
            Get started: Select a field layout and click Run!
            """)
        return
    
    # Run scan if button clicked
    if run_clicked and clf is not None:
        with st.spinner("🔄 Drone scanning field..."):
            progress_bar = st.progress(0, text="Starting scan...")
            status_text = st.empty()
            
            def update_progress(progress):
                progress_bar.progress(progress, text=f"Scanning... {int(progress * 100)}%")
                if progress > 0.5:
                    status_text.info(f"🔍 Analyzing disease patterns... {int(progress * 100)}%")
            
            try:
                scan_results = run_scan(
                    field_data=st.session_state.field_data,
                    spread_steps=spread_steps,
                    crop_type=st.session_state.crop_type,
                    clf=clf,
                    progress_callback=update_progress
                )
                
                st.session_state.scan_results = scan_results
                st.session_state.metrics = scan_results['metrics']
                st.session_state.scan_complete = True
                
                progress_bar.empty()
                status_text.empty()
                
                st.success("✅ Scan complete! Results displayed below.")
                st.balloons()
                
            except Exception as e:
                st.error(f"❌ Error during scan: {e}")
                import traceback
                st.code(traceback.format_exc())
                return
    
    # DISPLAY RESULTS
    if st.session_state.scan_complete and st.session_state.scan_results is not None:
        results = st.session_state.scan_results
        metrics = st.session_state.metrics
        
        # Section 1: Metric Cards
        display_metrics(metrics)
        st.divider()
        
        # Section 2: Heatmap
        col_grid, col_legend = st.columns([2, 1])
        
        with col_grid:
            st.subheader("🗺️ Field Health Map")
            fig = create_heatmap(results['results'])
            st.plotly_chart(fig, width='stretch', key="heatmap")
        
        with col_legend:
            st.subheader("📖 Legend")
            for status, color in [
                ("🟢 Healthy", "#2ecc71"),
                ("🟡 Early Disease", "#f1c40f"),
                ("🔴 Severe Disease", "#e74c3c"),
                ("⬛ Obstacle", "#34495e"),
                ("⬜ Unscanned", "#ecf0f1")
            ]:
                st.markdown(f'<span style="display:inline-block; width:20px; height:20px; background-color:{color}; border-radius:4px;"></span> {status}', unsafe_allow_html=True)
            
            st.divider()
            st.markdown("**📊 NDVI Color Scale**")
            st.caption("🟢 Higher values = Healthier crops")
            st.caption("🔴 Lower values = Disease stress")
        
        st.divider()
        
        # Section 3: Charts
        col_pie, col_bar = st.columns(2)
        
        with col_pie:
            pie_fig = create_pie_chart(metrics)
            st.plotly_chart(pie_fig, width='stretch')
        
        with col_bar:
            bar_fig = create_bar_chart(metrics)
            st.plotly_chart(bar_fig, width='stretch')
        
        # Section 4: AI Field Report
        st.divider()
        st.subheader("🤖 AI Field Report")
        
        col_report, col_actions = st.columns([3, 1])
        
        with col_actions:
            report_btn = st.button("📄 Generate Report", width='stretch')
            spray_btn = st.button("💊 Get Spray Advice", width='stretch')
            
            if metrics:
                df = pd.DataFrame({
                    'Metric': ['Scanned', 'Healthy', 'Early Disease', 'Severe Disease', 'Obstacles'],
                    'Count': [metrics['scanned'], metrics['healthy'], metrics['early'], metrics['severe'], metrics['obstacles']],
                    'Percentage': [
                        f"{metrics['scanned']/metrics['total_cells']*100:.1f}%",
                        f"{metrics['healthy_pct']:.1f}%",
                        f"{metrics['early_pct']:.1f}%",
                        f"{metrics['severe_pct']:.1f}%",
                        f"{metrics['obstacles']/metrics['total_cells']*100:.1f}%"
                    ]
                })
                csv = df.to_csv(index=False)
                st.download_button(
                    label="⬇️ Download CSV",
                    data=csv,
                    file_name=f"field_report_{st.session_state.field_name}.csv",
                    mime="text/csv",
                    width='stretch'
                )
        
        with col_report:
            if report_btn:
                with st.spinner("🤖 Generating report with AI..."):
                    report_text = generate_report(
                        metrics=metrics,
                        crop_type=st.session_state.crop_type,
                        field_name=st.session_state.field_name,
                        disease_seeds=st.session_state.field_data.get('disease_seeds', [])
                    )
                st.session_state.report_text = report_text
            
            if spray_btn:
                with st.spinner("🤖 Generating spray advice..."):
                    report_text = get_spray_advice(
                        metrics=metrics,
                        crop_type=st.session_state.crop_type,
                        disease_seeds=st.session_state.field_data.get('disease_seeds', [])
                    )
                st.session_state.report_text = report_text
            
            if st.session_state.report_text:
                st.markdown(
                    f'<div class="report-text">{st.session_state.report_text}</div>',
                    unsafe_allow_html=True
                )
            else:
                st.info("💡 Click 'Generate Report' or 'Get Spray Advice' to see AI recommendations.")
        
        # Section 5: Feature Importance
        with st.expander("📊 ML Model Explainability - Feature Importance"):
            if clf is not None:
                importances = clf.feature_importances_
                feature_names = ['NDVI', 'Red Intensity', 'Green Intensity', 'Texture Variance', 'Moisture Index']
                
                fig_feature = go.Figure(data=[
                    go.Bar(
                        x=feature_names,
                        y=importances,
                        marker_color='#2e7d32',
                        text=[f"{imp:.3f}" for imp in importances],
                        textposition='auto'
                    )
                ])
                
                fig_feature.update_layout(
                    title="Feature Importance in Disease Classification",
                    xaxis_title="Feature",
                    yaxis_title="Importance",
                    height=300,
                    margin=dict(l=20, r=20, t=40, b=40)
                )
                
                st.plotly_chart(fig_feature, width='stretch')
                
                st.markdown("**How the model makes decisions:**")
                for name, imp in zip(feature_names, importances):
                    st.markdown(f"- **{name}**: {imp:.2%} importance")

if __name__ == "__main__":
    main()
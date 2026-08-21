"""
app.py - Main Streamlit Application for AgriDrone
Modern UI Design
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
from datetime import datetime

# Import modules
from config import GRID_SIZE
from core_logic import run_scan
from llm_handler import init_openai, generate_report, get_spray_advice
# Page configuration
st.set_page_config(
    page_title="AgriDrone - Smart Crop Monitoring",
    layout="wide",
    page_icon="🌿",
    initial_sidebar_state="expanded"
)

# ============================================
st.markdown("""
<style>
    /* Main background */
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #e8edf2 100%);
    }
    
    /* Header styling */
    .main-header {
        font-size: 2.8rem;
        font-weight: 800;
        background: linear-gradient(135deg, #1a472a, #2e7d32, #43a047);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        padding: 10px 0;
        letter-spacing: -0.5px;
    }
    
    .sub-header {
        font-size: 1rem;
        color: #546e7a;
        font-weight: 400;
        margin-top: -10px;
        margin-bottom: 20px;
    }
    
    /* Card styling */
    .metric-card {
        background: white;
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        border: 1px solid rgba(46, 125, 50, 0.1);
        transition: transform 0.2s;
        text-align: center;
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 15px rgba(0,0,0,0.08);
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1a472a;
    }
    
    .metric-label {
        font-size: 0.9rem;
        color: #78909c;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #1a472a 0%, #2e7d32 100%);
    }
    
    .sidebar-title {
        color: white !important;
        font-size: 1.5rem !important;
        font-weight: 700 !important;
        padding: 10px 0 !important;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #2e7d32, #43a047);
        color: white;
        border: none;
        padding: 12px 24px;
        border-radius: 10px;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s;
        width: 100%;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(46, 125, 50, 0.4);
        background: linear-gradient(135deg, #1b5e20, #2e7d32);
    }
    
    .stButton > button:disabled {
        background: #bdbdbd;
        cursor: not-allowed;
        transform: none;
    }
    
    /* Report styling */
    .report-container {
        background: white;
        padding: 25px;
        border-radius: 15px;
        border-left: 5px solid #2e7d32;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
        margin: 10px 0;
    }
    
    .report-text {
        font-family: 'Segoe UI', sans-serif;
        line-height: 1.8;
        color: #263238;
        font-size: 0.95rem;
    }
    
    /* Legend styling */
    .legend-item {
        display: flex;
        align-items: center;
        padding: 6px 0;
        font-size: 0.9rem;
    }
    
    .legend-color {
        width: 20px;
        height: 20px;
        border-radius: 6px;
        margin-right: 12px;
        flex-shrink: 0;
    }
    
    /* Divider */
    .custom-divider {
        height: 2px;
        background: linear-gradient(90deg, transparent, #2e7d32, transparent);
        margin: 20px 0;
        opacity: 0.3;
    }
    
    /* Info box */
    .info-box {
        background: #e3f2fd;
        padding: 20px;
        border-radius: 12px;
        border-left: 4px solid #1976d2;
        margin: 10px 0;
    }
    
    /* Progress bar */
    .stProgress > div > div {
        background: linear-gradient(90deg, #2e7d32, #66bb6a) !important;
    }
    
    /* Toggle and select styling */
    .stSelectbox > div, .stSlider > div {
        background: transparent;
    }
</style>
""", unsafe_allow_html=True)

# ============================================
# SESSION STATE INITIALIZATION
# ============================================
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

# ============================================
# HELPER FUNCTIONS
# ============================================
@st.cache_resource
def load_model():
    """Load the trained Random Forest model."""
    model_path = Path('models/disease_clf.pkl')
    if model_path.exists():
        return joblib.load(model_path)
    else:
        st.error("⚠️ Model not found. Please run disease_model.py first.")
        return None

def create_default_fields():
    """Create default field configuration files."""
    data_dir = Path('data')
    data_dir.mkdir(exist_ok=True)
    
    fields = {
        "flat_farm": {
            "name": "Open Field - Wheat",
            "crop_type": "Wheat",
            "grid_configuration": {"rows": GRID_SIZE, "cols": GRID_SIZE, "drone_start": [0, 0]},
            "obstacles": [],
            "disease_seeds": [
                {"cell": [5, 5], "type": "early"},
                {"cell": [12, 18], "type": "early"},
                {"cell": [20, 7], "type": "severe"},
                {"cell": [8, 20], "type": "early"},
                {"cell": [15, 15], "type": "severe"}
            ]
        },
        "pond_farm": {
            "name": "Pond Field - Cotton",
            "crop_type": "Cotton",
            "grid_configuration": {"rows": GRID_SIZE, "cols": GRID_SIZE, "drone_start": [0, 0]},
            "obstacles": [[10, 10], [10, 11], [11, 10], [11, 11]],
            "disease_seeds": [
                {"cell": [5, 5], "type": "early"},
                {"cell": [18, 3], "type": "severe"},
                {"cell": [3, 18], "type": "early"}
            ]
        },
        "dense_field": {
            "name": "Dense Field - Rice",
            "crop_type": "Rice",
            "grid_configuration": {"rows": GRID_SIZE, "cols": GRID_SIZE, "drone_start": [0, 0]},
            "obstacles": [[7, 7], [7, 8], [8, 7]],
            "disease_seeds": [
                {"cell": [3, 3], "type": "early"},
                {"cell": [20, 20], "type": "severe"},
                {"cell": [12, 12], "type": "early"},
                {"cell": [5, 20], "type": "severe"},
                {"cell": [20, 5], "type": "early"}
            ]
        }
    }
    
    for name, data in fields.items():
        with open(data_dir / f'{name}.json', 'w') as f:
            json.dump(data, f, indent=2)
    
    return list(fields.keys())

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
    class_names = {0: "Healthy", 1: "Early Disease", 2: "Severe Disease", -1: "Obstacle", -2: "Unscanned"}
    
    # Create hover text
    hover_text = np.empty((rows, cols), dtype=object)
    for r in range(rows):
        for c in range(cols):
            val = results[r, c]
            status = class_names.get(val, "Unknown")
            hover_text[r, c] = f"<b>Location</b>: ({r}, {c})<br><b>Status</b>: {status}"
    
    fig = go.Figure(data=go.Heatmap(
        z=results,
        colorscale=[
            [0, "#ecf0f1"],   # Unscanned
            [0.25, "#34495e"], # Obstacle
            [0.5, "#2ecc71"],  # Healthy
            [0.75, "#f1c40f"], # Early Disease
            [1.0, "#e74c3c"]   # Severe Disease
        ],
        text=hover_text,
        hoverinfo='text',
        showscale=True,
        zmin=-2,
        zmax=2,
        colorbar=dict(
            title="Status",
            tickvals=[-2, -1, 0, 1, 2],
            ticktext=["Unscanned", "Obstacle", "Healthy", "Early", "Severe"]
        )
    ))
    
    fig.update_layout(
        title=dict(
            text="🌾 Field Health Map",
            font=dict(size=20, color="#1a472a")
        ),
        height=550,
        xaxis=dict(
            title="Column",
            showgrid=False,
            tickmode='linear',
            dtick=5,
            tickfont=dict(size=10)
        ),
        yaxis=dict(
            title="Row",
            showgrid=False,
            autorange='reversed',
            tickmode='linear',
            dtick=5,
            tickfont=dict(size=10)
        ),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=60, r=60, t=60, b=60)
    )
    
    return fig

def create_pie_chart(metrics):
    """Create modern pie chart."""
    labels = ['Healthy', 'Early Disease', 'Severe Disease']
    values = [metrics['healthy'], metrics['early'], metrics['severe']]
    colors = ['#2ecc71', '#f1c40f', '#e74c3c']
    
    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        marker=dict(colors=colors, line=dict(color='white', width=2)),
        textinfo='label+percent',
        textposition='outside',
        hole=0.35,
        pull=[0, 0.02, 0.05],
        rotation=90
    )])
    
    fig.update_layout(
        title=dict(
            text="📊 Disease Distribution",
            font=dict(size=18, color="#1a472a")
        ),
        height=380,
        margin=dict(l=30, r=30, t=50, b=30),
        showlegend=False,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig

def create_bar_chart(metrics):
    """Create modern bar chart."""
    categories = ['Healthy', 'Early Disease', 'Severe Disease']
    values = [metrics['healthy'], metrics['early'], metrics['severe']]
    colors = ['#2ecc71', '#f1c40f', '#e74c3c']
    
    fig = go.Figure(data=[go.Bar(
        x=categories,
        y=values,
        marker_color=colors,
        text=values,
        textposition='auto',
        textfont=dict(size=14, weight='bold'),
        width=0.6
    )])
    
    fig.update_layout(
        title=dict(
            text="📈 Disease Counts",
            font=dict(size=18, color="#1a472a")
        ),
        xaxis_title="Category",
        yaxis_title="Number of Cells",
        height=380,
        margin=dict(l=30, r=30, t=50, b=30),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(gridcolor='rgba(0,0,0,0.05)'),
        yaxis=dict(gridcolor='rgba(0,0,0,0.05)')
    )
    
    return fig

def display_metrics(metrics):
    """Display modern metric cards."""
    cols = st.columns(4)
    
    metric_configs = [
        (cols[0], "📊 Cells Scanned", metrics['scanned'], f"{metrics['scanned']}/{metrics['total_cells']}"),
        (cols[1], "✅ Healthy", metrics['healthy'], f"{metrics['healthy_pct']:.1f}%"),
        (cols[2], "⚠️ Early Disease", metrics['early'], f"{metrics['early_pct']:.1f}%"),
        (cols[3], "🚨 Severe Disease", metrics['severe'], f"{metrics['severe_pct']:.1f}%")
    ]
    
    for col, label, value, delta in metric_configs:
        with col:
            st.markdown(f"""
                <div class="metric-card">
                    <div style="font-size:0.8rem; color:#78909c; font-weight:500; text-transform:uppercase; letter-spacing:0.5px;">
                        {label}
                    </div>
                    <div class="metric-value">{value}</div>
                    <div style="font-size:0.85rem; color:#546e7a; margin-top:5px;">
                        {delta}
                    </div>
                </div>
            """, unsafe_allow_html=True)

# ============================================
# MAIN APPLICATION
# ============================================
def main():
    """Main app function."""
    clf = load_model()
    
    # ─── HEADER ───────────────────────────────────────
    col_logo, col_title = st.columns([1, 5])
    with col_logo:
        st.markdown("""
            <div style="font-size: 4rem; text-align: center; line-height: 1;">
                🌿
            </div>
        """, unsafe_allow_html=True)
    with col_title:
        st.markdown('<div class="main-header">AgriDrone</div>', unsafe_allow_html=True)
        st.markdown('<div class="sub-header">Smart Crop Health Monitoring & Disease Detection System</div>', unsafe_allow_html=True)
    
    # ─── SIDEBAR ──────────────────────────────────────
    with st.sidebar:
        st.markdown('<div class="sidebar-title">🌿 Control Panel</div>', unsafe_allow_html=True)
        st.markdown("---")
        
        # Crop selection
        crop_type = st.selectbox(
            "🌾 Crop Type",
            ["Wheat", "Cotton", "Rice", "Sugarcane"],
            index=0
        )
        st.session_state.crop_type = crop_type
        
        # Field selection
        field_options = load_field_options()
        field_choice = st.selectbox(
            "🗺️ Field Layout",
            field_options
        )
        
        # File uploader
        with st.expander("📤 Upload Custom Field"):
            uploaded_file = st.file_uploader(
                "Choose JSON file",
                type=["json"]
            )
            if uploaded_file is not None:
                try:
                    field_data = json.load(uploaded_file)
                    if 'grid_configuration' in field_data:
                        field_data['rows'] = field_data['grid_configuration'].get('rows', GRID_SIZE)
                        field_data['cols'] = field_data['grid_configuration'].get('cols', GRID_SIZE)
                    else:
                        field_data['rows'] = GRID_SIZE
                        field_data['cols'] = GRID_SIZE
                    st.session_state.field_data = field_data
                    st.session_state.field_name = field_data.get('name', 'Custom Field')
                    st.session_state.crop_type = field_data.get('crop_type', crop_type)
                    st.success("✅ Loaded successfully!")
                except Exception as e:
                    st.error(f"Error: {e}")
        
        # Load selected field
        if field_choice and not uploaded_file:
            try:
                field_path = Path('data') / f'{field_choice}.json'
                if field_path.exists():
                    with open(field_path, 'r') as f:
                        field_data = json.load(f)
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
                pass
        
        st.markdown("---")
        
        # Show current field info
        if st.session_state.field_data:
            st.info(f"📍 {st.session_state.field_name}\n\n🌾 {st.session_state.crop_type}")
        
        st.markdown("---")
        
        # Disease spread
        spread_steps = st.slider(
            "🔄 Disease Spread",
            min_value=0,
            max_value=10,
            value=5,
            help="Higher values = more disease spread"
        )
        
        st.markdown("---")
        
        # Run button
        run_clicked = st.button(
            "🚀 Start Scan",
            type="primary",
            use_container_width=True,
            disabled=clf is None
        )
        
        st.markdown("---")
        
        # API Status
        # API Status - OpenAI
        api_key = os.getenv('OPENAI_API_KEY')
        if api_key and init_openai(api_key):
            st.success("✅ OpenAI API Ready")
        else:
            st.warning("⚠️ OpenAI API Not Configured")
            st.caption("Add OPENAI_API_KEY to .env")
            st.caption("Get key from: https://platform.openai.com/api-keys")
    
    # ─── MAIN CONTENT ──────────────────────────────────
    
    # If no field loaded
    if st.session_state.field_data is None:
        st.markdown("""
            <div class="info-box">
                <h4>🚁 Welcome to AgriDrone</h4>
                <p>Select a field layout from the sidebar and click <b>Start Scan</b> to begin monitoring your crops.</p>
                <p style="margin-top:10px; font-size:0.9rem; color:#455a64;">
                    💡 The drone will scan a 25x25 grid, detect diseases using AI, and generate a comprehensive report.
                </p>
            </div>
        """, unsafe_allow_html=True)
        
        # Quick start cards
        cols = st.columns(3)
        quick_fields = ["🌾 Flat Farm", "💧 Pond Farm", "🌿 Dense Field"]
        for i, (col, name) in enumerate(zip(cols, quick_fields)):
            with col:
                st.markdown(f"""
                    <div style="
                        background: white;
                        padding: 20px;
                        border-radius: 15px;
                        text-align: center;
                        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
                        border: 1px solid rgba(46,125,50,0.1);
                    ">
                        <div style="font-size: 2.5rem;">{['🌾','💧','🌿'][i]}</div>
                        <div style="font-weight: 600; margin: 10px 0;">{name}</div>
                        <div style="font-size:0.85rem; color:#78909c;">Select from sidebar</div>
                    </div>
                """, unsafe_allow_html=True)
        return
    
    # Run scan
    if run_clicked and clf is not None:
        with st.spinner("🚀 Deploying drone..."):
            progress_bar = st.progress(0, text="Initializing scan...")
            status_text = st.empty()
            
            def update_progress(progress):
                progress_bar.progress(progress, text=f"Scanning... {int(progress * 100)}%")
                if progress > 0.5:
                    status_text.info(f"🔍 Analyzing crop health... {int(progress * 100)}%")
            
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
                
                st.success("✅ Scan Complete! Results displayed below.")
                st.balloons()
                
            except Exception as e:
                st.error(f"❌ Error: {e}")
                return
    
    # ─── DISPLAY RESULTS ──────────────────────────────
    if st.session_state.scan_complete and st.session_state.scan_results is not None:
        results = st.session_state.scan_results
        metrics = st.session_state.metrics
        
        # Section 1: Metrics
        display_metrics(metrics)
        st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
        
        # Section 2: Heatmap
        st.markdown("### 🗺️ Field Health Map")
        col_grid, col_legend = st.columns([3, 1])
        
        with col_grid:
            fig = create_heatmap(results['results'])
            st.plotly_chart(fig, use_container_width=True)
        
        with col_legend:
            st.markdown("#### 📖 Legend")
            legend_items = [
                ("🟢 Healthy", "#2ecc71"),
                ("🟡 Early Disease", "#f1c40f"),
                ("🔴 Severe Disease", "#e74c3c"),
                ("⬛ Obstacle", "#34495e"),
                ("⬜ Unscanned", "#ecf0f1")
            ]
            for label, color in legend_items:
                st.markdown(f"""
                    <div class="legend-item">
                        <div class="legend-color" style="background:{color};"></div>
                        {label}
                    </div>
                """, unsafe_allow_html=True)
            
            st.markdown("---")
            st.markdown("#### 📊 Field Summary")
            st.markdown(f"""
                <div style="font-size:0.9rem; color:#546e7a;">
                    <b>Total:</b> {metrics['total_cells']} cells<br>
                    <b>Scanned:</b> {metrics['scanned']} cells<br>
                    <b>Healthy:</b> {metrics['healthy']} cells<br>
                    <b>Affected:</b> {metrics['early'] + metrics['severe']} cells
                </div>
            """, unsafe_allow_html=True)
        
        st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
        
        # Section 3: Charts
        col_pie, col_bar = st.columns(2)
        with col_pie:
            pie_fig = create_pie_chart(metrics)
            st.plotly_chart(pie_fig, use_container_width=True)
        with col_bar:
            bar_fig = create_bar_chart(metrics)
            st.plotly_chart(bar_fig, use_container_width=True)
        
        st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
        
        # Section 4: AI Report
        st.markdown("### 🤖 AI Field Report")
        
        col_actions, col_content = st.columns([1, 3])
        
        with col_actions:
            report_btn = st.button("📄 Generate Report", use_container_width=True)
            spray_btn = st.button("💊 Spray Advice", use_container_width=True)
            
            if metrics:
                df = pd.DataFrame({
                    'Metric': ['Scanned', 'Healthy', 'Early', 'Severe', 'Obstacles'],
                    'Count': [
                        metrics['scanned'],
                        metrics['healthy'],
                        metrics['early'],
                        metrics['severe'],
                        metrics['obstacles']
                    ],
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
                    label="⬇️ Download Report",
                    data=csv,
                    file_name=f"agridrone_report_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
        
        with col_content:
            disease_seeds = st.session_state.field_data.get('disease_seeds', [])
            
            if report_btn:
                with st.spinner("🤖 Generating report..."):
                    report = generate_report(
                        metrics=metrics,
                        crop_type=st.session_state.crop_type,
                        field_name=st.session_state.field_name,
                        disease_seeds=disease_seeds
                    )
                st.session_state.report_text = report
            
            if spray_btn:
                with st.spinner("🤖 Generating spray advice..."):
                    report = get_spray_advice(
                        metrics=metrics,
                        crop_type=st.session_state.crop_type,
                        disease_seeds=disease_seeds
                    )
                st.session_state.report_text = report
            
            if st.session_state.report_text:
                st.markdown(f"""
                    <div class="report-container">
                        <div class="report-text">{st.session_state.report_text}</div>
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.info("💡 Click 'Generate Report' or 'Spray Advice' to get AI recommendations.")
        
        # Section 5: Model Explainability
        with st.expander("📊 AI Model Insights - Feature Importance"):
            if clf is not None:
                importances = clf.feature_importances_
                feature_names = ['NDVI', 'Red Intensity', 'Green Intensity', 'Texture Variance', 'Moisture Index']
                
                fig_feature = go.Figure(data=[
                    go.Bar(
                        x=feature_names,
                        y=importances,
                        marker_color='#2e7d32',
                        text=[f"{imp:.1%}" for imp in importances],
                        textposition='auto'
                    )
                ])
                
                fig_feature.update_layout(
                    title="How the AI Makes Decisions",
                    xaxis_title="Feature",
                    yaxis_title="Importance",
                    height=300,
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    xaxis=dict(gridcolor='rgba(0,0,0,0.05)'),
                    yaxis=dict(gridcolor='rgba(0,0,0,0.05)')
                )
                
                st.plotly_chart(fig_feature, use_container_width=True)
                
                st.markdown("**How the model detects disease:**")
                for name, imp in zip(feature_names, importances):
                    bar_length = int(imp * 30)
                    st.markdown(f"""
                        <div style="margin:5px 0;">
                            <span style="font-weight:500;">{name}</span>
                            <div style="background:#e0e0e0; border-radius:5px; height:8px; margin-top:3px;">
                                <div style="background:#2e7d32; width:{imp*100}%; height:8px; border-radius:5px;"></div>
                            </div>
                            <span style="font-size:0.8rem; color:#78909c;">{imp:.1%} importance</span>
                        </div>
                    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
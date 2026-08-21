@echo off
echo ========================================
echo     AgriDrone Setup Script
echo ========================================
echo.

echo Step 1: Creating virtual environment...
python -m venv venv
echo.

echo Step 2: Activating virtual environment...
call venv\Scripts\activate
echo.

echo Step 3: Upgrading pip...
python -m pip install --upgrade pip
echo.

echo Step 4: Installing requirements...
pip install streamlit==1.36.0
pip install plotly==5.23.0
pip install scikit-learn==1.5.0
pip install numpy==1.26.4
pip install pandas==2.2.0
pip install google-generativeai==0.7.0
pip install python-dotenv==1.0.0
pip install joblib==1.4.2
pip install protobuf==4.25.9
echo.

echo Step 5: Creating folders...
python -c "import os; [os.makedirs(f, exist_ok=True) for f in ['models','data','results']]"
echo.

echo Step 6: Creating .env file...
echo GEMINI_API_KEY=your_api_key_here > .env
echo.

echo Step 7: Creating field data...
python create_fields.py
echo.

echo Step 8: Training model...
python disease_model.py
echo.

echo ========================================
echo     Setup Complete!
echo ========================================
echo.
echo To run the app:
echo 1. Activate venv: venv\Scripts\activate
echo 2. Run: streamlit run app.py
echo.
pause
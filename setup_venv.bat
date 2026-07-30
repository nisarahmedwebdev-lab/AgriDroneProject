@echo off
echo Creating virtual environment...
python -m venv venv

echo Activating virtual environment...
call venv\Scripts\activate

echo Upgrading pip...
python -m pip install --upgrade pip

echo Installing packages...
pip install numpy==1.24.3
pip install pandas==2.0.3
pip install scikit-learn==1.3.0
pip install joblib==1.3.2
pip install python-dotenv==1.0.0
pip install protobuf==3.20.3
pip install streamlit==1.28.0
pip install plotly==5.18.0
pip install google-generativeai==0.3.2

echo Setup complete!
echo To activate the environment, run: venv\Scripts\activate
pause
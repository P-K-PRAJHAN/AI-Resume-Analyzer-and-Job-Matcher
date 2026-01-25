@echo off
echo Starting AI Resume Analyzer & Job Matcher...
echo.
echo Make sure Ollama is running with a model (e.g., 'ollama run phi3')
echo Press Ctrl+C to stop the application
echo.
cd /d "%~dp0"
python -m streamlit run app.py
pause
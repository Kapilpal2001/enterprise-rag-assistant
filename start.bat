@echo off
echo Starting Enterprise RAG Voice Assistant...
cd /d "%~dp0"
streamlit run app.py
pause

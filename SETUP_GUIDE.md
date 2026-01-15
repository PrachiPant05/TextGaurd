# TextGaurd - Setup and Run Guide

## Overview
TextGaurd is a Streamlit-based writing assistant application that provides:
- Text Analysis (AI detection, perplexity, burstiness)
- Grammar Checking
- Text Summarization
- Paraphrasing
- AI Article Writer

## Prerequisites
- Python 3.8 or higher
- Google API Key (for Gemini AI)
- Hugging Face API Token (optional, for summarization and paraphrasing)

## Step-by-Step Setup Instructions

### 1. Install Python Dependencies

If you have a virtual environment already set up (like the `TextGaurd` folder), activate it first:

**Windows (PowerShell):**
```powershell
.\TextGaurd\Scripts\Activate.ps1
```

**Windows (Command Prompt):**
```cmd
TextGaurd\Scripts\activate.bat
```

**If you need to create a new virtual environment:**
```powershell
python -m venv TextGaurd
.\TextGaurd\Scripts\Activate.ps1
```

### 2. Install Required Packages

```powershell
pip install -r requirements.txt
```

This will install:
- streamlit
- crewai
- langchain_google_genai
- nltk
- language-tool-python
- matplotlib
- and other dependencies

### 3. Set Up Environment Variables

Create a `.env` file in the project root directory with the following:

```env
GOOGLE_API_KEY=your_google_api_key_here
HUGGINGFACE_API_TOKEN=your_huggingface_token_here
GEMINI_MODEL=gemini-pro
```

**Optional: GEMINI_MODEL** - If you encounter model not found errors, you can specify which Gemini model to use:
- `gemini-pro` (default, most widely available)
- `gemini-1.5-pro` (if available in your region)
- `gemini-1.5-flash` (if available in your region)

**How to get API keys:**
- **Google API Key**: 
  1. Go to https://makersuite.google.com/app/apikey
  2. Create a new API key
  3. Copy and paste it into your `.env` file

- **Hugging Face Token** (optional):
  1. Go to https://huggingface.co/settings/tokens
  2. Create a new token
  3. Copy and paste it into your `.env` file
  4. Note: Some features (summarization, paraphrasing) require this token

### 4. Download NLTK Data

The application will automatically download NLTK data on first run, but you can also download it manually:

```python
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab'); nltk.download('stopwords'); nltk.download('brown')"
```

## Running the Application

### Start the Streamlit App

Make sure you're in the project root directory and your virtual environment is activated, then run:

```powershell
streamlit run app.py
```

The application will start and automatically open in your default web browser at `http://localhost:8501`

If it doesn't open automatically, you can manually navigate to the URL shown in the terminal.

## Troubleshooting

### Common Issues:

1. **"GOOGLE_API_KEY not found" error**
   - Make sure you've created a `.env` file in the project root
   - Verify the API key is correct and has no extra spaces
   - Restart the Streamlit app after creating/updating the `.env` file

2. **NLTK data download errors**
   - The app will try to download NLTK data automatically
   - If it fails, manually download using the command in step 4 above

3. **Port already in use**
   - If port 8501 is busy, Streamlit will try the next available port
   - Check the terminal output for the actual URL

4. **Module not found errors**
   - Make sure your virtual environment is activated
   - Reinstall dependencies: `pip install -r requirements.txt`

5. **Hugging Face API errors**
   - Some models may take 20-30 seconds to load on first use
   - If you see 503 errors, wait a moment and try again
   - The HUGGINGFACE_API_TOKEN is optional but required for summarization and paraphrasing features

## Features

Once the app is running, you can access:

1. **Text Analysis**: Analyze text for AI-generated patterns
2. **Grammar Check**: Check grammar and spelling errors
3. **Text Summarization**: Condense long articles (requires Hugging Face token)
4. **Paraphrasing**: Rewrite text while maintaining meaning (requires Hugging Face token)
5. **Article Writer**: Generate articles on any topic using AI (requires Google API key)

## Stopping the Application

Press `Ctrl+C` in the terminal where Streamlit is running to stop the application.


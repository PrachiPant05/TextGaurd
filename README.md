
# TextGaurd - AI-Powered Writing Assistant

A comprehensive writing assistant application built with Streamlit, CrewAI, and advanced NLP models. TextGaurd helps you analyze text, check grammar, summarize content, detect AI-generated text, and write compelling articles.

## Features

### 1. Text Analysis
- Calculate text perplexity and burstiness scores
- Detect AI-generated content patterns
- Visualize most common and repeated words
- Statistical analysis of writing patterns

### 2. Grammar Check
- Real-time grammar and spelling corrections
- Detailed error explanations
- Context-aware suggestions
- Powered by LanguageTool

### 3. Text Summarization
- Condense long articles into concise summaries
- BART model for high-quality abstractive summarization
- Maintain key information and context

### 4. Article Writer
- AI-powered article generation using Google Gemini
- Research-backed content creation
- CrewAI agents for comprehensive research and writing
- Download articles as Markdown files

### 5. Paraphrasing (Coming Soon)
- Rewrite text while maintaining meaning
- T5-based paraphrasing engine

### 6. Plagiarism Check (Coming Soon)
- Text similarity detection
- Source attribution

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager
- Google Gemini API key - [Get it here](https://makersuite.google.com/app/apikey)
- Serper API key - [Get it here](https://serper.dev/)

### Step 1: Clone the Repository
```bash
git clone <your-repo-url>
cd TextGaurd-main
```

### Step 2: Create Virtual Environment
```bash
python -m venv TextGaurd
```

### Step 3: Activate Virtual Environment

**Windows:**
```bash
TextGaurd\Scripts\activate
```

**Mac/Linux:**
```bash
source TextGaurd/bin/activate
```

### Step 4: Install Dependencies
```bash
pip install --upgrade pip
pip install streamlit nltk language-tool-python transformers torch torchvision torchaudio crewai crewai-tools langchain-google-genai python-dotenv matplotlib pandas numpy
```

### Step 5: Download NLTK Data
```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('brown'); nltk.download('punkt_tab')"
```

### Step 6: Configure API Keys

Create a `.env` file in the project root:

```env
GOOGLE_API_KEY="your_google_gemini_api_key_here"
SERPER_API_KEY="your_serper_api_key_here"
```

**Important:** Never commit your `.env` file to version control. Add it to `.gitignore`.

## Usage

### Running the Streamlit Application

```bash
streamlit run app.py
```

The application will open in your browser at `http://localhost:8501`

### Running CrewAI Standalone

To run the research and writing agents directly:

```bash
python crew.py
```

This will generate an article and save it to `DemoOutput.md`

## How to Use Each Feature

### Text Analysis
1. Select "Text Analysis" from the sidebar
2. Paste your text in the text area
3. Click "Analyze"
4. View perplexity, burstiness scores, and visualizations

### Grammar Check
1. Select "Grammar Check" from the sidebar
2. Enter or paste your text
3. Click "Check Grammar"
4. Review errors, suggestions, and context

### Text Summarization
1. Select "Text Summarization" from the sidebar
2. Paste the text you want to summarize
3. Click "Summarize"
4. Get a concise summary maintaining key points

### Article Writer
1. Select "Article Writer" from the sidebar
2. Enter your topic (e.g., "Artificial Intelligence in Healthcare")
3. Click "Write Article"
4. Download the generated article as a Markdown file

## Project Structure

```
TextGaurd-main/
├── TextGaurd/              # Virtual environment
├── .env                    # API keys (not in repo)
├── app.py                  # Main Streamlit application
├── agents.py               # CrewAI agent definitions
├── tasks.py                # CrewAI task definitions
├── crew.py                 # CrewAI orchestration
├── tools.py                # Search tool configuration
├── requirements.txt        # Python dependencies
├── README.md              # This file
└── DemoOutput.md          # Generated article output
```

## Configuration

### Model Configuration

The application uses these pre-trained models:

- **BART** (`facebook/bart-large-cnn`) - Text summarization
- **T5** (`t5-base`) - Text paraphrasing
- **Gemini 1.5 Flash** - Article generation and research

### Customizing Agents

Edit `agents.py` to modify:
- Agent roles and goals
- Temperature and creativity settings
- Backstories and behavior

### Customizing Tasks

Edit `tasks.py` to modify:
- Task descriptions
- Expected output formats
- Research depth

## Technical Details

### AI Models Used
- **Language Model**: Google Gemini 1.5 Flash
- **Summarization**: BART (Facebook AI)
- **Paraphrasing**: T5 (Google)
- **Grammar**: LanguageTool
- **Web Search**: Serper API

### Technologies
- **Frontend**: Streamlit
- **AI Framework**: CrewAI
- **NLP**: Transformers, NLTK
- **LLM Integration**: LangChain
- **Visualization**: Matplotlib

## Important Notes

### First Run
The first time you run the application, it will download large AI models (approximately 2.5 GB total):
- BART model: approximately 1.6 GB
- T5 model: approximately 850 MB

This download happens automatically and takes 5-15 minutes depending on your internet speed.

### Memory Requirements
The transformer models require significant RAM (recommended: 8GB+). If you encounter memory errors, close other applications or use a machine with more RAM.

### API Rate Limits
- Free tier API keys have usage limits
- Monitor your usage on Google AI Studio and Serper dashboards
- Consider upgrading for heavy usage

## Troubleshooting

### "Module not found" errors
```bash
pip install <missing-module-name>
```

### Version conflicts
```bash
pip install transformers --upgrade
pip install tokenizers --upgrade
```

### Grammar check not working
Ensure Java is installed (required by language-tool-python):
- Download from [java.com](https://www.java.com/)

### Models not downloading
- Check your internet connection
- Ensure you have enough disk space (approximately 3GB free)
- Try manually downloading from Hugging Face

## Security

- Never commit `.env` files to version control
- Add `.env` to your `.gitignore` file
- Rotate API keys if accidentally exposed
- Use environment variables for production deployments

## Future Enhancements

- [ ] Complete paraphrasing feature UI
- [ ] Implement plagiarism detection
- [ ] Add user authentication
- [ ] Create article history/database
- [ ] Support multiple languages
- [ ] Export to multiple formats (PDF, DOCX)
- [ ] Real-time collaboration features
- [ ] Browser extension
- [ ] API endpoint for integrations

## Acknowledgments

- Google Gemini for powerful language understanding
- Hugging Face for pre-trained models
- CrewAI for agent orchestration
- Streamlit for the web framework
- The open-source community

![image](https://github.com/user-attachments/assets/07ed5099-2b44-495c-977e-d4b6ec97f50f)

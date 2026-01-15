import streamlit as st
import nltk
import os
import json
from dotenv import load_dotenv

# Load environment variables first
load_dotenv()

# Set a dummy OPENAI_API_KEY before importing CrewAI to prevent import errors
# This won't be used since we explicitly pass Gemini LLM to agents
if 'OPENAI_API_KEY' not in os.environ:
    os.environ['OPENAI_API_KEY'] = 'dummy-key-not-used'

from crewai import Agent
from tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
import requests
import signal
import logging
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError

# Initialize the LLM with the Google API key
# Updated to use gemini-1.5-flash (gemini-pro is deprecated)
google_api_key = os.getenv("GOOGLE_API_KEY")
if not google_api_key:
    raise ValueError("GOOGLE_API_KEY not found in environment variables. Please set it in your .env file.")

# Initialize Gemini LLM using Google Generative AI SDK directly for better compatibility
# This avoids langchain wrapper issues with model names
try:
    import google.generativeai as genai
    genai.configure(api_key=google_api_key)
    
    # Try to get available models
    available_models = []
    try:
        for model in genai.list_models():
            if 'generateContent' in model.supported_generation_methods:
                model_name = model.name.replace('models/', '')
                available_models.append(model_name)
    except Exception:
        # If listing fails, use common defaults
        available_models = ["gemini-1.5-flash", "gemini-1.5-pro", "gemini-pro"]
    
    # Try to initialize with available models
    llm = None
    model_names_to_try = available_models[:3] if available_models else ["gemini-1.5-flash", "gemini-1.5-pro", "gemini-pro"]
    
    for model_name in model_names_to_try:
        try:
            # Use langchain wrapper with the model name
            llm = ChatGoogleGenerativeAI(
                model=model_name,
                verbose=True,
                temperature=0.5,
                google_api_key=google_api_key
            )
            # Test if it works by trying to invoke (but don't actually call)
            break
        except Exception as e:
            if "404" not in str(e) and "NOT_FOUND" not in str(e):
                # Non-404 error, might be a different issue
                raise
            continue
    
    if llm is None:
        # Fallback: use langchain without specifying model (uses default)
        llm = ChatGoogleGenerativeAI(
            verbose=True,
            temperature=0.5,
            google_api_key=google_api_key
        )
        
except ImportError:
    # If google.generativeai is not available, use langchain directly
    gemini_model = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
    llm = ChatGoogleGenerativeAI(
        model=gemini_model,
        verbose=True,
        temperature=0.5,
        google_api_key=google_api_key
    )
except Exception as e:
    # Final fallback
    raise ValueError(
        f"Failed to initialize Gemini LLM.\n"
        f"Error: {str(e)}\n"
        f"Please check:\n"
        f"1. Your GOOGLE_API_KEY is valid\n"
        f"2. Gemini API is enabled in Google Cloud Console\n"
        f"3. Your API key has access to Gemini models\n"
        f"Try installing: pip install google-generativeai"
    )

# IMPORTANT: We set a dummy OPENAI_API_KEY above to satisfy CrewAI's import requirements
# All agents explicitly use the Gemini LLM (llm parameter), so OpenAI is never actually used
# CrewAI will use the explicitly passed LLM, not the dummy OpenAI key

# Hugging Face API setup
# Updated to use working model endpoints
HF_API_TOKEN = os.getenv("HUGGINGFACE_API_TOKEN")
# Using working models (updated to avoid deprecated endpoints)
HF_API_URL_SUMMARIZATION = "https://api-inference.huggingface.co/models/facebook/bart-large-cnn"
HF_API_URL_PARAPHRASE = "https://api-inference.huggingface.co/models/prithivida/parrot_paraphraser_on_T5"

headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}

# Download NLTK resources
@st.cache_resource
def download_nltk_data():
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', quiet=True)
        nltk.download('punkt_tab', quiet=True)
        nltk.download('stopwords', quiet=True)
        nltk.download('brown', quiet=True)

download_nltk_data()

from nltk.util import ngrams
from nltk.lm.preprocessing import padded_everygram_pipeline
from nltk.lm import MLE
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
import string

# Cache the language model training - this is expensive and should only happen once
@st.cache_resource
def get_language_model():
    """Train and cache the language model. This only runs once."""
    try:
        # Use a sample of Brown corpus for faster training (first 50k words)
        tokens = list(nltk.corpus.brown.words())[:50000]
        train_data, padded_vocab = padded_everygram_pipeline(1, tokens)
        model = MLE(1)
        model.fit(train_data, padded_vocab)
        return model
    except Exception as e:
        st.error(f"Error training language model: {str(e)}")
        return None


# Limit input size for text analysis
MAX_TEXT_LENGTH = 1000  # Maximum number of characters

# Add a timeout for long-running computations
TIMEOUT_SECONDS = 10  # Reduced since we use simpler calculation now

def safe_execute(func, *args, **kwargs):
    """Execute a function with a timeout."""
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(func, *args, **kwargs)
        try:
            return future.result(timeout=TIMEOUT_SECONDS)
        except TimeoutError:
            return None

def preprocess_text(text):
    if len(text) > MAX_TEXT_LENGTH:
        text = text[:MAX_TEXT_LENGTH]
    tokens = nltk.word_tokenize(text.lower())
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words and token not in string.punctuation]
    return tokens


def plot_most_common_words(text):
    try:
        tokens = preprocess_text(text)
        if not tokens:
            st.info("No words found in the text after preprocessing.")
            return
        
        word_freq = nltk.FreqDist(tokens)
        most_common_words = word_freq.most_common(10)

        if not most_common_words:
            st.info("No common words found.")
            return

        words, counts = zip(*most_common_words)

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(words, counts)
        ax.set_xlabel('Words')
        ax.set_ylabel('Frequency')
        ax.set_title('Most Common Words')
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)  # Close to free memory
    except Exception as e:
        st.warning(f"Could not generate word frequency plot: {str(e)}")


def plot_repeated_words(text):
    try:
        tokens = preprocess_text(text)
        if not tokens:
            st.info("No words found in the text after preprocessing.")
            return
            
        word_freq = nltk.FreqDist(tokens)
        repeated_words = [word for word, count in word_freq.items() if count > 1][:10]

        # Check if there are any repeated words
        if not repeated_words:
            st.info("No repeated words found in the text.")
            return

        words, counts = zip(*[(word, word_freq[word]) for word in repeated_words])

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(words, counts)
        ax.set_xlabel('Words')
        ax.set_ylabel('Frequency')
        ax.set_title('Repeated Words')
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)  # Close to free memory
    except Exception as e:
        st.warning(f"Could not generate repeated words plot: {str(e)}")


def calculate_text_complexity(text):
    """Calculate a simpler text complexity metric as an alternative to perplexity.
    This is much faster and doesn't require a language model."""
    tokens = preprocess_text(text)
    if not tokens or len(tokens) < 2:
        return None
    
    try:
        # Calculate vocabulary diversity (unique words / total words)
        unique_words = len(set(tokens))
        total_words = len(tokens)
        vocabulary_diversity = unique_words / total_words if total_words > 0 else 0
        
        # Calculate average word length
        avg_word_length = sum(len(word) for word in tokens) / len(tokens) if tokens else 0
        
        # Calculate repetition ratio (how often words repeat)
        word_freq = nltk.FreqDist(tokens)
        repeated_count = sum(1 for count in word_freq.values() if count > 1)
        repetition_ratio = repeated_count / len(word_freq) if len(word_freq) > 0 else 0
        
        # Combine metrics into a complexity score (0-100 scale)
        # Lower diversity and higher repetition = lower complexity (more AI-like)
        # Higher diversity and lower repetition = higher complexity (more human-like)
        complexity_score = (vocabulary_diversity * 50) + ((1 - repetition_ratio) * 50)
        
        # Normalize to a scale similar to perplexity (higher = more complex/human-like)
        # Convert to a perplexity-like metric (inverse relationship)
        perplexity_estimate = max(50, 200 - (complexity_score * 1.5))
        
        return perplexity_estimate
    except Exception as e:
        return None

def calculate_perplexity(text, model):
    """Calculate perplexity with improved error handling - fallback to simpler method."""
    # Try the simple method first (much faster)
    simple_perplexity = calculate_text_complexity(text)
    if simple_perplexity is not None:
        return simple_perplexity
    
    # Fallback to model-based calculation if simple method fails
    tokens = preprocess_text(text)
    if not tokens or len(tokens) < 2:
        return None
    
    try:
        # Limit the number of tokens to prevent timeout
        if len(tokens) > 50:  # Further reduced
            tokens = tokens[:50]
        
        padded_tokens = ['<s>'] + tokens + ['</s>']
        ngrams_sequence = list(ngrams(padded_tokens, model.order))
        if not ngrams_sequence:
            return simple_perplexity  # Return simple estimate
        
        # Calculate perplexity with error handling
        perplexity = model.perplexity(ngrams_sequence)
        
        # Handle infinity or invalid values
        if perplexity == float('inf') or perplexity != perplexity:
            return simple_perplexity  # Return simple estimate
        
        # Ensure perplexity is within reasonable bounds
        if perplexity < 0 or perplexity > 1e10:
            return simple_perplexity  # Return simple estimate
            
        return perplexity
    except (ValueError, ZeroDivisionError, OverflowError, TimeoutError):
        return simple_perplexity  # Return simple estimate
    except Exception:
        return simple_perplexity  # Return simple estimate


def calculate_burstiness(text):
    tokens = preprocess_text(text)
    word_freq = nltk.FreqDist(tokens)

    avg_freq = sum(word_freq.values()) / len(word_freq)
    variance = sum((freq - avg_freq) ** 2 for freq in word_freq.values()) / len(word_freq)

    burstiness_score = variance / (avg_freq ** 2)
    return burstiness_score


# Safe wrapper functions for timeout handling
def calculate_perplexity_safe(text, model):
    """Calculate perplexity with timeout protection."""
    return safe_execute(calculate_perplexity, text, model)


def calculate_burstiness_safe(text):
    """Calculate burstiness with timeout protection."""
    return safe_execute(calculate_burstiness, text)


def is_generated_text(perplexity, burstiness_score):
    """Determine if text is likely AI-generated based on metrics."""
    if perplexity is None or burstiness_score is None:
        return "Unable to determine (insufficient data)"
    
    # Adjusted thresholds for the new complexity metric
    # Lower perplexity (complexity) and lower burstiness = more AI-like
    if perplexity < 120 and burstiness_score < 1:
        return "Likely generated by a language model"
    elif perplexity < 150 and burstiness_score < 1.5:
        return "Possibly generated by a language model"
    else:
        return "Not likely generated by a language model"


# Grammar Checking Function using Gemini AI
def check_grammar_with_explanations(text):
    """Check grammar using Gemini AI - no Java required."""
    if not text or len(text.strip()) == 0:
        return []
    
    try:
        # Create a prompt for Gemini to check grammar
        prompt = f"""You are an expert English grammar and spelling checker. Analyze the following text and identify ALL grammar, spelling, and punctuation errors.

For each error found, provide:
1. Error type (e.g., "Grammar error", "Spelling mistake", "Punctuation error", "Word choice")
2. The exact incorrect text/phrase
3. Suggested corrections (as an array)
4. Brief explanation or context

Text to check:
"{text}"

IMPORTANT: Return ONLY a valid JSON array. Each error object must have these exact fields:
- "error": string describing the error type and issue
- "incorrect_text": string of the exact text that has the error
- "suggestions": array of strings with suggested corrections
- "context": string with brief explanation or surrounding context

If no errors are found, return an empty array: []

Example format:
[
  {{
    "error": "Subject-verb agreement error",
    "incorrect_text": "HE ARE",
    "suggestions": ["He is", "They are"],
    "context": "Subject 'HE' (singular) requires singular verb 'is', not 'are'"
  }}
]

Return ONLY the JSON array, no other text:"""
        
        response = llm.invoke(prompt)
        
        # Extract content from response
        if hasattr(response, 'content'):
            response_text = response.content
        elif hasattr(response, 'text'):
            response_text = response.text
        else:
            response_text = str(response)
        
        # Clean the response - remove markdown code blocks if present
        response_text = response_text.strip()
        if response_text.startswith('```'):
            # Remove markdown code blocks
            lines = response_text.split('\n')
            response_text = '\n'.join(lines[1:-1]) if len(lines) > 2 else response_text
        elif response_text.startswith('```json'):
            lines = response_text.split('\n')
            response_text = '\n'.join(lines[1:-1]) if len(lines) > 2 else response_text
        
        # Try to extract JSON from the response
        try:
            # Find JSON array in the response
            json_start = response_text.find('[')
            json_end = response_text.rfind(']') + 1
            if json_start >= 0 and json_end > json_start:
                json_text = response_text[json_start:json_end]
                errors = json.loads(json_text)
                
                # Validate and format errors
                formatted_errors = []
                for error in errors:
                    if isinstance(error, dict):
                        suggestions = error.get('suggestions', [])
                        if not isinstance(suggestions, list):
                            suggestions = [suggestions] if suggestions else []
                        
                        formatted_errors.append({
                            'error': error.get('error', 'Grammar issue'),
                            'incorrect_text': error.get('incorrect_text', ''),
                            'suggestions': suggestions,
                            'context': error.get('context', '')
                        })
                return formatted_errors
        except (json.JSONDecodeError, ValueError) as e:
            # If JSON parsing fails, check if response indicates no errors
            response_lower = response_text.lower()
            if any(phrase in response_lower for phrase in ['no error', 'no issues', 'no mistakes', 'correct', 'perfect']):
                return []
            
            # If we can't parse JSON but got a response, try to extract useful info
            # This is a fallback for when Gemini doesn't return perfect JSON
            if len(response_text) > 50:  # Likely has content
                # Try to find error mentions
                if 'error' in response_lower or 'mistake' in response_lower or 'incorrect' in response_lower:
                    return [{
                        'error': 'Grammar check completed (parsing issue)',
                        'incorrect_text': '',
                        'suggestions': [],
                        'context': response_text[:300]  # Show first 300 chars
                    }]
        
        return []
        
    except Exception as e:
        error_msg = str(e)
        if "404" in error_msg or "NOT_FOUND" in error_msg:
            st.error("⚠️ Gemini API error: Model not available. Please check your GOOGLE_API_KEY and ensure Gemini API is enabled in Google Cloud Console.")
        else:
            st.warning(f"Grammar checker encountered an error: {error_msg}")
        return []


# Fallback functions using Gemini when Hugging Face models are unavailable
def summarize_with_gemini(text):
    """Summarize text using Gemini as fallback."""
    try:
        prompt = f"Please provide a concise summary of the following text:\n\n{text[:2000]}"
        response = llm.invoke(prompt)
        if hasattr(response, 'content'):
            return response.content
        elif hasattr(response, 'text'):
            return response.text
        return str(response)
    except Exception as e:
        error_msg = str(e)
        if "404" in error_msg or "NOT_FOUND" in error_msg:
            return "Error: Gemini model not available. Please check your GOOGLE_API_KEY and ensure Gemini API is enabled in Google Cloud Console."
        return f"Error during summarization: {error_msg}"

def paraphrase_with_gemini(text):
    """Paraphrase text using Gemini as fallback."""
    try:
        prompt = f"Please paraphrase the following text while maintaining its meaning:\n\n{text[:1000]}"
        response = llm.invoke(prompt)
        if hasattr(response, 'content'):
            return response.content
        elif hasattr(response, 'text'):
            return response.text
        return str(response)
    except Exception as e:
        error_msg = str(e)
        if "404" in error_msg or "NOT_FOUND" in error_msg:
            return "Error: Gemini model not available. Please check your GOOGLE_API_KEY and ensure Gemini API is enabled in Google Cloud Console. You may need to enable the Gemini API in your Google Cloud project."
        return f"Error during paraphrasing: {error_msg}"


# Text Summarization using Hugging Face API
def summarize_text(input_text):
    """Summarize text using Hugging Face API."""
    if not HF_API_TOKEN:
        return "Error: Hugging Face API token not configured. Please set HUGGINGFACE_API_TOKEN in your .env file."
    
    try:
        payload = {
            "inputs": input_text[:1024],  # Limit input length
            "parameters": {
                "max_length": 150,
                "min_length": 30,
                "do_sample": False
            }
        }
        response = requests.post(
            HF_API_URL_SUMMARIZATION,
            headers=headers,
            json=payload,
            timeout=60  # Increased timeout for model loading
        )
        
        # Handle 503 (model loading) or 410 (gone) errors
        if response.status_code == 503:
            return "Error: Model is currently loading. Please wait a moment and try again. Hugging Face models may take 20-30 seconds to load on first use."
        elif response.status_code == 410:
            # Try fallback to Gemini for summarization
            return summarize_with_gemini(input_text)
        
        response.raise_for_status()
        result = response.json()
        
        # Handle different response formats
        if isinstance(result, list) and len(result) > 0:
            if isinstance(result[0], dict):
                return result[0].get('summary_text', result[0].get('generated_text', 'Summary generation failed.'))
            else:
                return str(result[0])
        elif isinstance(result, dict):
            return result.get('summary_text', result.get('generated_text', 'Summary generation failed.'))
        else:
            return "Error: Unexpected response format from API."
    except requests.exceptions.RequestException as e:
        return f"Error calling Hugging Face API: {str(e)}"
    except Exception as e:
        return f"Error during summarization: {str(e)}"


# Paraphrasing using Hugging Face API
def paraphrase_text(input_text):
    """Paraphrase text using Hugging Face API."""
    if not HF_API_TOKEN:
        return "Error: Hugging Face API token not configured. Please set HUGGINGFACE_API_TOKEN in your .env file."
    
    try:
        payload = {
            "inputs": input_text[:512],  # Limit input length
            "parameters": {
                "max_length": 150,
                "num_beams": 4,
                "early_stopping": True
            }
        }
        response = requests.post(
            HF_API_URL_PARAPHRASE,
            headers=headers,
            json=payload,
            timeout=30
        )
        
        # Handle 410 (gone) errors - use Gemini as fallback
        if response.status_code == 410:
            return paraphrase_with_gemini(input_text)
        
        response.raise_for_status()
        result = response.json()
        
        if isinstance(result, list) and len(result) > 0:
            if isinstance(result[0], dict) and 'generated_text' in result[0]:
                return result[0]['generated_text']
            elif isinstance(result[0], str):
                return result[0]
        elif isinstance(result, dict):
            if 'generated_text' in result:
                return result['generated_text']
            elif 'paraphrased_text' in result:
                return result['paraphrased_text']
        
        return "Error: Unexpected response format from API."
    except requests.exceptions.RequestException as e:
        return f"Error calling Hugging Face API: {str(e)}"
    except Exception as e:
        return f"Error during paraphrasing: {str(e)}"


# Configure logging
logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)

try:
    # Verify Gemini API key is available
    if not os.getenv("GOOGLE_API_KEY"):
        st.error("GOOGLE_API_KEY not found in environment variables. Please set it in your .env file.")
        st.stop()
    
    # Initialize article writer using Gemini directly (avoiding CrewAI OpenAI fallback issues)
    # We'll use Gemini LLM directly for article generation instead of CrewAI Agent
    # This avoids the issue where CrewAI tries to use OpenAI despite passing Gemini LLM
    pass
except ImportError as e:
    logger.error("Failed to initialize LLM: %s", e)
    st.error("The application failed to initialize the language model. Please check your configuration and dependencies.")
    st.stop()
except Exception as e:
    logger.error("Failed to initialize Agent: %s", e)
    st.error(f"The application failed to initialize the agent: {str(e)}")
    st.exception(e)
    st.stop()


# Disable signal handling for `crewai`
def disable_signal_handlers():
    def noop_handler(*args, **kwargs):
        pass

    signal.signal = noop_handler

disable_signal_handlers()

# Streamlit App Logic
def main():
    st.title("TextGaurd - Your Writing Assistant")
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    option = st.sidebar.selectbox("Select Feature", [
        "Text Analysis", 
        "Grammar Check", 
        "Paraphrasing", 
        "Text Summarization", 
        "Article Writer"
    ])
    
    if option == "Text Analysis":
        st.header("Text Analysis")
        st.write("Analyze your text for AI-generated patterns, perplexity, and burstiness.")
        
        text = st.text_area("Enter the text you want to analyze", height=200)
        
        if st.button("Analyze"):
            if text:
                with st.spinner("Analyzing text... (This may take a few seconds on first run)"):
                    # Get cached language model (only trains once)
                    model = get_language_model()
                    
                    if model is None:
                        st.error("Failed to initialize language model. Please try again.")
                        st.stop()

                    # Calculate perplexity/complexity
                    perplexity = calculate_perplexity_safe(text, model)
                    if perplexity is None:
                        st.warning("Text complexity calculation could not be computed.")
                        perplexity_display = "N/A"
                    else:
                        perplexity_display = f"{perplexity:.2f}"
                        st.metric("Text Complexity Score", perplexity_display, 
                                help="Higher scores indicate more complex/varied text (more human-like). Lower scores suggest simpler, more repetitive text (potentially AI-generated).")

                    # Calculate burstiness score
                    burstiness_score = calculate_burstiness_safe(text)
                    if burstiness_score is None:
                        st.warning("Burstiness calculation timed out.")
                        burstiness_display = "N/A"
                    else:
                        burstiness_display = f"{burstiness_score:.2f}"
                        st.metric("Burstiness Score", burstiness_display,
                                help="Measures word frequency distribution. Lower scores (<1) may indicate AI-generated text with more uniform word usage.")

                    # Check if text is likely generated by a language model
                    if perplexity is not None and burstiness_score is not None:
                        generated_cue = is_generated_text(perplexity, burstiness_score)
                        st.info(f"**Analysis Result:** {generated_cue}")
                    elif perplexity is not None:
                        # Can still provide analysis with just complexity score
                        if perplexity < 120:
                            st.info("**Analysis Result:** Text shows lower complexity (possibly AI-generated)")
                        else:
                            st.info("**Analysis Result:** Text shows higher complexity (likely human-written)")
                    elif burstiness_score is not None:
                        # Can still provide analysis with just burstiness
                        if burstiness_score < 1:
                            st.info("**Analysis Result:** Low burstiness detected (possibly AI-generated)")
                        else:
                            st.info("**Analysis Result:** Higher burstiness detected (likely human-written)")
                    else:
                        st.info("**Analysis Result:** Unable to determine (metrics could not be calculated)")

                    # Plot most common words
                    st.subheader("Most Common Words")
                    plot_most_common_words(text)

                    # Plot repeated words
                    st.subheader("Repeated Words")
                    plot_repeated_words(text)
            else:
                st.warning("Please enter some text to analyze.")

    elif option == "Grammar Check":
        st.header("Grammar Check")
        st.write("Check your text for grammar and spelling errors.")
        
        text = st.text_area("Enter text for grammar check:", "", height=200)
        
        if st.button("Check Grammar"):
            if text:
                with st.spinner("Checking grammar..."):
                    grammar_errors = check_grammar_with_explanations(text)
                    
                    if grammar_errors:
                        st.warning(f"Found {len(grammar_errors)} grammar issue(s)")
                        
                        for i, error in enumerate(grammar_errors, 1):
                            with st.expander(f"Error {i}: {error['error']}"):
                                st.write(f"**Incorrect Text:** {error['incorrect_text']}")
                                # Ensure suggestions is a list
                                suggestions = error.get('suggestions', [])
                                if not isinstance(suggestions, list):
                                    suggestions = [suggestions] if suggestions else []
                                if suggestions:
                                    st.write(f"**Suggestions:** {', '.join(str(s) for s in suggestions[:3])}")
                                else:
                                    st.write("**Suggestions:** No suggestions available")
                                st.write(f"**Context:** {error.get('context', 'N/A')}")
                    else:
                        st.success("No grammar issues detected!")
            else:
                st.warning("Please enter some text to check.")

    elif option == "Text Summarization":
        st.header("Text Summarization")
        st.write("Condense long articles into concise summaries.")
        
        text = st.text_area("Enter text to summarize:", "", height=200)
        
        if st.button("Summarize"):
            if text:
                with st.spinner("Summarizing... (This may take 20-30 seconds on first use)"):
                    summary = summarize_text(text)
                    st.subheader("Summary")
                    st.write(summary)
            else:
                st.warning("Please enter some text to summarize.")

    elif option == "Paraphrasing":
        st.header("Paraphrasing")
        st.write("Rewrite your text while maintaining its meaning.")
        
        text = st.text_area("Enter text to paraphrase:", "", height=200)
        
        if st.button("Paraphrase"):
            if text:
                with st.spinner("Paraphrasing... (This may take 20-30 seconds on first use)"):
                    paraphrased = paraphrase_text(text)
                    st.subheader("Paraphrased Text")
                    st.write(paraphrased)
            else:
                st.warning("Please enter some text to paraphrase.")

    elif option == "Article Writer":
        st.header("AI Article Writer")
        st.write("Generate compelling articles on any topic using AI.")
        
        topic = st.text_input("Enter the topic you want to explore:", "")
        
        if st.button("Write Article"):
            if topic:
                with st.spinner("Generating article... This may take a minute."):
                    try:
                        # Generate the article using Gemini directly (avoiding CrewAI issues)
                        prompt = f"""Write a compelling, well-structured tech article about {topic}. 
                        The article should be informative, engaging, and suitable for a general tech audience.
                        Include an introduction, main content with key points, and a conclusion.
                        Make it approximately 500-800 words."""
                        
                        response = llm.invoke(prompt)
                        
                        # Extract content from response
                        if hasattr(response, 'content'):
                            article_content = response.content
                        elif hasattr(response, 'text'):
                            article_content = response.text
                        else:
                            article_content = str(response)
                        
                        st.subheader("Generated Article")
                        st.write(article_content)

                        # Prepare Markdown content for download
                        md_content = f"# {topic}\n\n{article_content}"
                        
                        # Download button
                        st.download_button(
                            label="Download Article as .md",
                            data=md_content,
                            file_name=f"{topic.replace(' ', '_')}.md",
                            mime="text/markdown"
                        )
                    except Exception as e:
                        st.error(f"Error generating article: {str(e)}")
                        st.exception(e)
            else:
                st.error("Please enter a topic to explore.")

    # Footer
    st.sidebar.markdown("---")
    st.sidebar.info("Built with Streamlit, CrewAI, and Hugging Face")


if __name__ == "__main__":
    main()
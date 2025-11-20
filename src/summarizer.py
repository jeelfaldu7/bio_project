import google.generativeai as genai
import os

# Load API key
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

MODEL_NAME = "gemini-1.5-flash"

def summarize_article(title: str, abstract: str) -> dict:
    prompt = f"""
    You are an AI biotech assistant. Summarize this article in 3 bullet points.
    Extract: 
    1. Main finding
    2. Key biological targets (genes, proteins, pathways)
    3. Application area (diagnostics, therapeutics, biotech tools, etc.)
    
    Title: {title}
    Abstract: {abstract}
    """

    response = genai.GenerativeModel(MODEL_NAME).generate_content(prompt)
    text = response.text

    return {
        "title": title,
        "raw_summary": abstract,
        "ai_summary": text
    }

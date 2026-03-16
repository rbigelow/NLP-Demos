import spacy
from spacy import displacy
import requests
from urllib.parse import urlparse
from bs4 import BeautifulSoup

# Load a small English NER model
try:
    nlp = spacy.load("en_core_web_sm")
except OSError as e:
    raise SystemExit(
        "Missing spaCy model 'en_core_web_sm'.\n"
        "Install it with:\n"
        "  python -m spacy download en_core_web_sm"
    ) from e

def run_ner(text):
    """
    Process text with spaCy NER and return entities.
    """
    doc = nlp(text)
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    return doc, entities


if __name__ == "__main__":
    
    print("=== Real-Time NER Demo ===")
    print("Enter a URL to extract and analyze text from a webpage.\n")
    
    url = input("Paste URL: ").strip()
    
    # Validate URL
    try:
        result = urlparse(url)
        if not all([result.scheme, result.netloc]):
            raise ValueError("Invalid URL format")
    except Exception as e:
        print(f"Error: Invalid URL - {e}")
        raise SystemExit(1)
    
    # Fetch webpage content
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
    except requests.RequestException as e:
        print(f"Error fetching URL: {e}")
        raise SystemExit(1)
    
    # Extract text from HTML
    try:
        soup = BeautifulSoup(response.content, "html.parser")
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()
        text = soup.get_text(separator="\n")
        text = "\n".join(line.strip() for line in text.split("\n") if line.strip())
    except ImportError:
        print("Error: beautifulsoup4 is required. Install with: pip install beautifulsoup4")
        raise SystemExit(1)
    
    doc, entities = run_ner(text)
    
    print("\n=== Extracted Entities ===")
    for ent, label in entities:
        print(f"{ent:30}  ->  {label}")
    
    # HTML visualization
    html = displacy.render(doc, style="ent", page=True)
    with open("ner_output.html", "w", encoding="utf-8") as f:
        f.write(html)
    
    print("\nA highlighted HTML visualization has been saved as ner_output.html")
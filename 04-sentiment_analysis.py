"""
Sentiment Analysis Demo
-----------------------
A simple classroom demonstration of sentiment analysis using NLTK's
VADER (Valence Aware Dictionary and sEntiment Reasoner) tool.

Install dependencies:
    pip install nltk

Run:
    python sentiment_analysis.py
"""

import nltk
from nltk.sentiment import SentimentIntensityAnalyzer


def download_dependencies():
    """Download the VADER lexicon if it isn't already available."""
    try:
        nltk.data.find("sentiment/vader_lexicon.zip")
    except LookupError:
        print("Downloading VADER lexicon...")
        nltk.download("vader_lexicon")


def classify_sentiment(compound_score: float) -> str:
    """Convert a compound score into a human-readable label."""
    if compound_score >= 0.05:
        return "Positive"
    elif compound_score <= -0.05:
        return "Negative"
    else:
        return "Neutral"


def analyze_text(analyzer: SentimentIntensityAnalyzer, text: str) -> None:
    """Analyze a single piece of text and print the results."""
    scores = analyzer.polarity_scores(text)
    sentiment = classify_sentiment(scores["compound"])

    print(f'\nText: "{text}"')
    print(f"  Negative: {scores['neg']:.3f}")
    print(f"  Neutral:  {scores['neu']:.3f}")
    print(f"  Positive: {scores['pos']:.3f}")
    print(f"  Compound: {scores['compound']:.3f}")
    print(f"  Sentiment: {sentiment}")


def run_demo():
    """Run the sentiment analysis demo on a set of example sentences."""
    download_dependencies()
    analyzer = SentimentIntensityAnalyzer()

    sample_texts = [
        "I absolutely love this course, it's amazing!",
        "This is the worst lecture I have ever attended.",
        "The weather today is okay, nothing special.",
        "I'm not sure how I feel about this assignment.",
        "What a fantastic demonstration of natural language processing!",
    ]

    print("=" * 60)
    print("Sentiment Analysis Demo (VADER)")
    print("=" * 60)

    for text in sample_texts:
        analyze_text(analyzer, text)

    print("\n" + "=" * 60)
    print("Try your own sentence! (type 'quit' to exit)")
    print("=" * 60)

    while True:
        user_input = input("\nEnter a sentence: ").strip()
        if user_input.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break
        if not user_input:
            continue
        analyze_text(analyzer, user_input)


if __name__ == "__main__":
    run_demo()

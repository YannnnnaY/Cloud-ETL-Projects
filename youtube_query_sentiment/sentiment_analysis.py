"""
method 1 - textblob
simple API and reliance on the popular Natural Language Toolkit (NLTK)
"""

from textblob import TextBlob
# https://textblob.readthedocs.io/en/dev/api_reference.html#textblob.blob.TextBlob.sentiment

# Function to compute sentiment
def get_sentiment(text):
    analysis = TextBlob(text)
    return analysis.sentiment.polarity

# Combine title and description into a single text field
def textblob_videos_sentiment(df):
    df['combined_text'] = df['title'] + ' ' + df['description']

    # Apply sentiment analysis on the combined text
    df['sentiment'] = df['combined_text'].apply(get_sentiment)

    # Add a sentiment label based on polarity
    df['sentiment_label'] = df['sentiment'].apply(lambda x: 'positive' if x > 0 else ('negative' if x < 0 else 'neutral'))

    return df


"""
method 2 - VADER (Valence Aware Dictionary and sEnti'ment Reasoner)
Specifically optimized for social media text, making it great for YouTube data.
"""

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Initialize VADER sentiment intensity analyzer
analyzer = SentimentIntensityAnalyzer()

# Function to classify sentiment based on VADER's compound score
def classify_vader_sentiment(compound_score):
    if compound_score >= 0.05:
        return 'positive'
    elif compound_score <= -0.05:
        return 'negative'
    else:
        return 'neutral'

# Apply sentiment analysis using VADER
def vader_videos_sentiment(df):
    df['combined_text'] = df['title'] + ' ' + df['description']
    df['vader_compound'] = df['combined_text'].apply(lambda text: analyzer.polarity_scores(text)['compound'])
    df['vader_sentiment'] = df['vader_compound'].apply(classify_vader_sentiment)

    # Preview the results with VADER sentiment
    return df

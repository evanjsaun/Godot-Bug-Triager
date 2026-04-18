import json 
import re
import nltk
import ssl
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation

NUM_TOPICS = 10

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# download NLTK data sets
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('punkt_tab', quiet=True)

ISSUES_FILE = "godot_issues.json"

CUSTOM_STOPWORDS = {"godot", "engine", "bug", 
                    "issue", "reproduce", "problem", 
                    "steps", "reproducible", "also", 
                    "like", "try", "still", 
                    "tested", "version", "verify",
                    "crash", "either", "think",
                    "normally", "let", "flag",
                    "see", "start", "end",
                    "work", "project", "game",
                    "edition", "build", "test"}

def clean_text(text):
    """
    Cleans the markdown input text and converts to usable format and reduces noise
    """
    if not text:
        return ""
    
    # remove any potential code blocks
    text = re.sub(r'```.*?```', '', text, flags=re.DOTALL)
    # remove urls
    text = re.sub(r'http\S+', '', text)
    # remove markdown formatting
    text = re.sub(r'[*_~`#]', '', text)

    return text

def preprocess_nlp(text):
    """
    Tokenizes and Lemmatizes text,
    Removes stop words,
    """
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))

    stop_words = stop_words.union(CUSTOM_STOPWORDS)

    # tokenize
    tokens = nltk.word_tokenize(text.lower())

    # lemmatize
    clean_tokens = []
    for word in tokens:
        # find alphabetic characters, skip stop words
        if word.isalpha() and word not in stop_words:
            # lemmatize suitable words
            clean_word = lemmatizer.lemmatize(word)
            clean_tokens.append(clean_word)
    return ' '.join(clean_tokens)

def process_issues(json_path):
    """
    Reads issues from path, processes and vectorizes text, then extracts topics using LDA
    """
    print(f"Processing issues from {json_path}...")

    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            issues = json.load(f)
    except FileNotFoundError:
        print(f"Error: File {json_path} not found.")
        return []
    
    processed_texts = []

    print(f"Cleaning and preprocessing {len(issues)} issues...")
    for issue in issues:
        # get title and content
        raw_text = f"{issue.get('title', '')} {issue.get('body', '')}"

        cleaned = clean_text(raw_text)
        processed = preprocess_nlp(cleaned)
        processed_texts.append(processed)

    print("Vectorizing text with TF-IDF...")
    # init vectorizer 
    # max_df = 0.95 ignores words that appear in >95% of issues
    # min_df = 2 ignores words that appear in <2 issues
    vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, stop_words='english')

    # fit and transform the processed text
    tfidf_matrix = vectorizer.fit_transform(processed_texts)
    print(f"Matrix Shape: {tfidf_matrix.shape} (issues x unique words)")

    print("Topic modelling and LDA...")
    lda_model = LatentDirichletAllocation(n_components=NUM_TOPICS, random_state=42)
    lda_model.fit(tfidf_matrix)

    # extract top words for each topic
    feature_names = vectorizer.get_feature_names_out()

    # print top words for each topic
    # components_ is a matrix (topics x words) where each value is the weight of that word in the topic 
    for topic_idx, topic in enumerate(lda_model.components_):
        print(f"\nTopic {topic_idx + 1}:")
        # top 10 weighted words
        # arg sort returns lowest -> highest, we want highest, so we take the last 10 and reverse
        top_words_idx = topic.argsort()[:-10 - 1:-1]
        top_words = [feature_names[i] for i in top_words_idx]
        print(", ".join(top_words))

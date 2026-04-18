import os
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

REPO_DIR = "./godot_repo"
TARGET_FOLDERS = ["core", "scene", "editor", "servers", "modules"]
TARGET_EXTENSIONS = [".cpp", ".h", ".gd", ".cs"]

def clean_text(text):
    """
    Cleans input text and converts specific case patterns to individual words to match with issue descriptions.
    Removes non-alphanumeric characters and extra whitespace.
    """
    # split snake case words
    text = text.replace('_', ' ')

    # split camel case words
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)

    # remove non-alphanumeric characters
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)

    # convert to lowercase and remove extra whitespace, and return
    return re.sub(r'\s+', ' ', text).strip().lower()

def extract_text_from_source(repo_dir):
    file_paths = []
    file_content = []

    print(f"Extracting text from source code in {repo_dir}...")
    for root, _, files in os.walk(repo_dir):
        if not any (folder in root for folder in TARGET_FOLDERS):
            continue

        for file in files:
            if file.endswith(tuple(TARGET_EXTENSIONS)):
                full_path = os.path.join(root, file)
                try:
                    with open(full_path, 'r', encoding='utf-8') as f:
                        content = f.read()

                    cleaned_content = clean_text(content)

                    if len(cleaned_content) > 10:
                        file_paths.append(full_path)
                        file_content.append(cleaned_content)
                except Exception as e:
                    print(f"Error reading {full_path}: {e}")
    print(f"Extracted text from {len(file_paths)} source files.")
    return file_paths, file_content

def get_files_for_issue(issue_text, file_paths, file_content, top_k=5):
    """
    Given an issue description, 
    returns the top_k most relevant source files based on 
    cosine similarity of TF-IDF vectors.
    """

    print("Initializing TF-IDF vectorizer for issue and source files...")

    # init and fit vectorizer
    vectorizer = TfidfVectorizer(stop_words='english', max_df=0.85, min_df=2)
    tfidf_matrix = vectorizer.fit_transform(file_content)

    # transform issue text into same vector space
    # reapply cleaning rules
    cleaned_issue = clean_text(issue_text)
    issue_vector = vectorizer.transform([cleaned_issue])

    # calculate similarity using cosine similarity
    print("Calculating cosine similarity between issue and source files...")
    similarities = cosine_similarity(issue_vector, tfidf_matrix).flatten()

    # use the same reverse sorting to get top_k indices of most similar files
    top_idx = similarities.argsort()[:-top_k-1:-1]

    print(f"Top {top_k} relevant files for the issue:")
    top_file_paths = []
    for idx in top_idx:
        score = similarities[idx]

        # only print files with some relevance
        if score > 0.05:
            clean_path = file_paths[idx].replace(REPO_DIR, "")
            # print and round similarity score float to 3 decimal places
            print(f"{clean_path} (similarity: {score:.3f})")
            top_file_paths.append(file_paths[idx])

    return top_file_paths
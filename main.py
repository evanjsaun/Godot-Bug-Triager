import os
import json
from github import Github
from github import Auth
from git import Repo

import process_issues
import process_source
import developer_recommender

# enter personal github token for faster download speeds
GITHUB_TOKEN = ""

REPO_NAME = "godotengine/godot"
CLONE_DIR = "./godot_repo"
ISSUES_FILE = "godot_issues.json"

MAX_ISSUES = 15

ISSUE_COLLECTION_LIMIT = 100

def clone_repo(repo_url, local_dir):
    """
    Clones the GitHub repository to the local directory.
    """
    print(f"Cloning repo...")
    if not os.path.exists(local_dir):
        print(f"Cloning repository {repo_url} into {local_dir}...")
        Repo.clone_from(repo_url, local_dir)
        print("Repository cloned successfully.")
    else:
        print(f"Directory {local_dir} already exists. Skipping clone.")

def fetch_issues(repo_name, token, limit):
    """
    Fetches issues from the specified GitHub repository.
    """
    print(f"Fetching issues from {repo_name}...")
    if token and token != "":
        auth = Auth.Token(token)
        g = Github(auth=auth)
    else:
        print("No token provided, please set the token variable for optimal download rates")
        g = Github()

    try: 
        repo = g.get_repo(repo_name)
        issues = repo.get_issues(state='closed')[:ISSUE_COLLECTION_LIMIT]

        issue_data = []
        count = 0

        print(f"Connected to {repo_name}. Fetching (up to {limit}) issues...")

        for issue in issues:
            # Filter PRs to get bug reports
            if issue.pull_request is None and any(label.name == "bug" for label in issue.labels):
                issue_data.append({
                    "number": issue.number,
                    "title": issue.title,
                    "body": issue.body,
                    "labels": [label.name for label in issue.labels],
                })
                count += 1

                # if count % 10 == 0:
                #     print(f"Fetched {count} issues...")

            if count >= limit:
                break

        return issue_data
    except Exception as e:
        print(f"Error fetching issues: {e}")
        return []

if __name__ == '__main__':
    # Clone the repository 
    repo_url = f"https://github.com/{REPO_NAME}.git"
    clone_repo(repo_url, CLONE_DIR)

    fetched_issues = fetch_issues(REPO_NAME, GITHUB_TOKEN, MAX_ISSUES)

    if fetched_issues:
        print("\n--- Saving Data ---")
        with open(ISSUES_FILE, 'w', encoding='utf-8') as f:
            json.dump(fetched_issues, f, indent=4)
        print(f"Saved {len(fetched_issues)} issues to {ISSUES_FILE}")

        # Process the issues
        process_issues.process_issues(ISSUES_FILE)

        # Extract text from source code
        paths, content = process_source.extract_text_from_source(CLONE_DIR)

        # Build file->author map once (loads from cache after first run)
        file_author_map = developer_recommender.build_file_author_map(CLONE_DIR)

        # recommend files and developers for each issue
        for issue in fetched_issues:
            issue_text = f"{issue['title']} {issue['body']}"
            print(f"\n--- Recommendations for Issue #{issue['number']}: {issue['title']} ---")

            top_files = process_source.get_files_for_issue(issue_text, paths, content, top_k=5)

            print("\n  Developer Recommendations:")
            developer_recommender.print_developer_recommendations(
                top_files,
                top_n=5,
                file_author_map=file_author_map,
            )

    else:
        print("No issues fetched. Exiting.\n")


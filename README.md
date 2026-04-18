# Bug-Triager

# Usage
- Install python on host machine

- If necessary, set up virtual environment
> python(3) -m venv .venv

> (Linux/MacOS) source .venv/bin/activate

> (Windows) .venv/Scripts/activate

- Install dependencies: 
> pip(3) install PyGithub GitPython nltk scikit-learn
- Generate personal access token (On GitLab: User Preferences -> Access -> Personal Access Tokens -> Add New Token)
- Replace GITHUB_TOKEN in main to your github token
- Adjust MAX_ISSUES and NUM_TOPICS to desired amounts
- Optionally add or change included stop words in process_issues.py/CUSTOM_STOPWORDS
- run:
> python(3) main.py
- wait to install godot repo on machine, let program run

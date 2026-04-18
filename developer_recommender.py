"""
developer_recommender.py

Given the file paths returned by process_source.get_files_for_issue(),
ranks developers who should fix the bug using an exponential decay
function on their commit history.

Decay formula (per commit):
    score = e^( -ln(2) / DECAY_HALF_LIFE_DAYS * days_since_commit )

A commit made DECAY_HALF_LIFE_DAYS ago is worth half a commit made today.
Scores are summed across all commits and all matched files per developer.
"""

import json
import math
import os
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from git import Repo
from tqdm import tqdm

# -----------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------

CLONE_DIR = "./godot_repo"       # must match main.py
DECAY_HALF_LIFE_DAYS = 180       # days until a commit's score halves
TOP_N_DEVELOPERS = 5             # developers to return per issue

FILE_AUTHOR_MAP_CACHE = "file_author_map.json"

SOURCE_EXTENSIONS = {".cpp", ".c", ".h", ".hpp", ".gd", ".gdscript", ".cs", ".py"}

_LAMBDA = math.log(2) / DECAY_HALF_LIFE_DAYS


# -----------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------

def recommend_developers(file_paths: list,
                         top_n: int = TOP_N_DEVELOPERS,
                         file_author_map: dict = None) -> list:
    """
    Rank developers for a bug report given its relevant source files.

    Parameters
    ----------
    file_paths      : list of file paths as returned by get_files_for_issue()
                      (full paths like "./godot_repo/scene/gui/button.cpp")
    top_n           : number of developers to return
    file_author_map : pre-built map; loaded from cache if None

    Returns
    -------
    List of dicts sorted by descending score:
        [{"rank", "author", "score", "commit_count", "files_matched", "most_recent"}, ...]
    """
    if file_author_map is None:
        file_author_map = _load_or_build_map()

    # Strip the repo directory prefix so paths match git's relative paths
    relative_paths = [_to_relative(p) for p in file_paths]

    now = datetime.now(tz=timezone.utc)
    author_scores  = defaultdict(float)
    author_commits = defaultdict(int)
    author_files   = defaultdict(set)
    author_latest  = {}

    for rel_path in relative_paths:
        for entry in file_author_map.get(rel_path, []):
            author    = entry["author"]
            commit_dt = _parse_iso(entry["date_iso"])
            days_ago  = max(0.0, (now - commit_dt).total_seconds() / 86_400)

            author_scores[author]  += math.exp(-_LAMBDA * days_ago)
            author_commits[author] += 1
            author_files[author].add(rel_path)

            if author not in author_latest or commit_dt > author_latest[author]:
                author_latest[author] = commit_dt

    ranked = sorted(author_scores.items(), key=lambda x: x[1], reverse=True)

    results = []
    for rank, (author, score) in enumerate(ranked[:top_n], start=1):
        results.append({
            "rank":          rank,
            "author":        author,
            "score":         round(score, 4),
            "commit_count":  author_commits[author],
            "files_matched": sorted(author_files[author]),
            "most_recent":   author_latest[author].isoformat() if author in author_latest else None,
        })

    return results


def print_developer_recommendations(file_paths: list,
                                    top_n: int = TOP_N_DEVELOPERS,
                                    file_author_map: dict = None) -> list:
    """
    Recommend developers and print the results.
    Returns the same list so callers can use the data too.
    """
    results = recommend_developers(file_paths, top_n=top_n, file_author_map=file_author_map)

    if not results:
        print("  No developer recommendations found for these files.")
        return results

    print(f"  Top {len(results)} recommended developer(s):")
    for dev in results:
        last = dev["most_recent"][:10] if dev["most_recent"] else "N/A"
        print(
            f"    #{dev['rank']} {dev['author']:<30}  "
            f"score={dev['score']:.3f}  "
            f"commits={dev['commit_count']}  "
            f"last active={last}"
        )
    return results


def build_file_author_map(clone_dir: str = CLONE_DIR,
                           max_commits: int = 30_000) -> dict:
    """
    Walk the git commit log and build:
        { "scene/gui/button.cpp": [{"author", "email", "date_iso", "sha"}, ...], ... }

    Saves to FILE_AUTHOR_MAP_CACHE. Run once; reloads from cache afterward.
    """
    print(f"[DeveloperRecommender] Building file->author map from {clone_dir}...")
    repo   = Repo(clone_dir)
    fa_map = defaultdict(list)

    commits = list(repo.iter_commits("HEAD", max_count=max_commits))
    for commit in tqdm(commits, desc="Mining commit history"):
        try:
            changed = _source_files_in_commit(commit)
        except Exception:
            continue

        author   = _resolve_author(commit)
        date_iso = datetime.fromtimestamp(
            commit.authored_date, tz=timezone.utc
        ).isoformat()

        for fpath in changed:
            fa_map[fpath].append({
                "author":   author,
                "email":    commit.author.email or "",
                "date_iso": date_iso,
                "sha":      commit.hexsha[:12],
            })

    # Sort newest-first within each file
    for fpath in fa_map:
        fa_map[fpath].sort(key=lambda x: x["date_iso"], reverse=True)

    with open(FILE_AUTHOR_MAP_CACHE, "w", encoding="utf-8") as f:
        json.dump(dict(fa_map), f, indent=2)

    print(f"[DeveloperRecommender] Map covers {len(fa_map)} files -> saved to {FILE_AUTHOR_MAP_CACHE}")
    return dict(fa_map)


# -----------------------------------------------------------------------
# Internal helpers
# -----------------------------------------------------------------------

def _load_or_build_map() -> dict:
    if os.path.exists(FILE_AUTHOR_MAP_CACHE):
        print("[DeveloperRecommender] Loading file->author map from cache.")
        with open(FILE_AUTHOR_MAP_CACHE, encoding="utf-8") as f:
            return json.load(f)
    return build_file_author_map()


def _to_relative(full_path: str) -> str:
    """
    Convert "./godot_repo/scene/gui/button.cpp" -> "scene/gui/button.cpp"
    so it matches the relative paths stored in git history.
    """
    p = Path(full_path)
    try:
        return str(p.relative_to(CLONE_DIR)).replace("\\", "/")
    except ValueError:
        return str(p).replace("\\", "/").lstrip("./")


def _source_files_in_commit(commit) -> list:
    if not commit.parents:
        return []
    diffs = commit.parents[0].diff(commit)
    return [
        d.b_path for d in diffs
        if d.b_path and Path(d.b_path).suffix.lower() in SOURCE_EXTENSIONS
    ]


def _resolve_author(commit) -> str:
    """Extract the most useful author identifier from a commit."""
    email = commit.author.email or ""
    name  = commit.author.name  or ""
    # GitHub no-reply emails encode the login: 12345678+username@users.noreply.github.com
    if "noreply.github.com" in email:
        part = email.split("@")[0]
        return part.split("+", 1)[1] if "+" in part else part
    return name or email.split("@")[0]


def _parse_iso(date_str: str) -> datetime:
    try:
        dt = datetime.fromisoformat(date_str)
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except ValueError:
        return datetime.now(tz=timezone.utc)

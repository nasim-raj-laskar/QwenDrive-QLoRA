import subprocess

def get_git_metadata():
    """Get minimal git metadata for experiment tracking."""
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True)
        branch = subprocess.run(["git", "rev-parse", "--abbrev-ref", "HEAD"], capture_output=True, text=True)
        status = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True)
        
        return {
            "git_commit": commit.stdout.strip()[:8] if commit.returncode == 0 else "unknown",
            "git_branch": branch.stdout.strip() if branch.returncode == 0 else "unknown",
            "git_is_dirty": bool(status.stdout.strip()) if status.returncode == 0 else True
        }
    except:
        return {"git_commit": "unknown", "git_branch": "unknown", "git_is_dirty": True}
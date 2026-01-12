"""GitHub repository cloning and processing service."""

import os
import shutil
import tempfile
import subprocess
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class RepoFile:
    """Represents a file in a repository."""
    path: str
    relative_path: str
    content: str
    size: int


class GitHubService:
    """Service for cloning and processing GitHub repositories."""
    
    def __init__(self, workspace_dir: str = None):
        self.workspace_dir = workspace_dir or tempfile.mkdtemp(prefix="migratemate_")
    
    def clone_repo(self, repo_url: str, branch: str = "main") -> str:
        """
        Clone a GitHub repository.
        
        Args:
            repo_url: GitHub URL (https://github.com/user/repo)
            branch: Branch to clone
            
        Returns:
            Path to cloned repository
        """
        # Clean URL
        repo_url = repo_url.strip()
        if repo_url.endswith('.git'):
            repo_url = repo_url[:-4]
        
        # Extract repo name
        repo_name = repo_url.rstrip('/').split('/')[-1]
        clone_path = os.path.join(self.workspace_dir, repo_name)
        
        # Remove if exists
        if os.path.exists(clone_path):
            shutil.rmtree(clone_path)
        
        # Clone
        logger.info(f"Cloning {repo_url} to {clone_path}")
        try:
            result = subprocess.run(
                ["git", "clone", "--depth", "1", "--branch", branch, repo_url, clone_path],
                capture_output=True,
                text=True,
                timeout=120
            )
            if result.returncode != 0:
                # Try without branch (use default)
                result = subprocess.run(
                    ["git", "clone", "--depth", "1", repo_url, clone_path],
                    capture_output=True,
                    text=True,
                    timeout=120
                )
                if result.returncode != 0:
                    raise Exception(f"Git clone failed: {result.stderr}")
        except subprocess.TimeoutExpired:
            raise Exception("Clone timeout - repository too large")
        
        return clone_path
    
    def find_python_files(self, repo_path: str, exclude_patterns: List[str] = None) -> List[str]:
        """Find all Python files in a repository."""
        exclude_patterns = exclude_patterns or [
            '__pycache__', '.git', 'venv', 'env', '.env',
            'node_modules', '.pytest_cache', '*.pyc', 'migrations',
            'tests', 'test_*', '*_test.py'
        ]
        
        python_files = []
        repo_path = Path(repo_path)
        
        for py_file in repo_path.rglob('*.py'):
            relative = py_file.relative_to(repo_path)
            
            # Check exclusions
            skip = False
            for pattern in exclude_patterns:
                if pattern in str(relative):
                    skip = True
                    break
            
            if not skip:
                python_files.append(str(py_file))
        
        return sorted(python_files)
    
    def read_file(self, file_path: str, repo_path: str) -> RepoFile:
        """Read a file and return RepoFile object."""
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        relative_path = os.path.relpath(file_path, repo_path)
        
        return RepoFile(
            path=file_path,
            relative_path=relative_path,
            content=content,
            size=len(content)
        )
    
    def detect_flask_project(self, repo_path: str) -> Dict[str, Any]:
        """Detect if repository is a Flask project and find entry points."""
        indicators = {
            'is_flask': False,
            'entry_points': [],
            'has_blueprints': False,
            'has_flask_restful': False,
            'config_files': []
        }
        
        python_files = self.find_python_files(repo_path)
        
        for file_path in python_files:
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                
                # Check for Flask imports
                if 'from flask import' in content or 'import flask' in content:
                    indicators['is_flask'] = True
                    
                    # Check for app creation
                    if 'Flask(__name__)' in content or 'Flask(' in content:
                        indicators['entry_points'].append(
                            os.path.relpath(file_path, repo_path)
                        )
                    
                    # Check for blueprints
                    if 'Blueprint(' in content:
                        indicators['has_blueprints'] = True
                    
                    # Check for Flask-RESTful
                    if 'flask_restful' in content or 'Api(' in content:
                        indicators['has_flask_restful'] = True
                
                # Config files
                basename = os.path.basename(file_path).lower()
                if basename in ['config.py', 'settings.py', 'configuration.py']:
                    indicators['config_files'].append(
                        os.path.relpath(file_path, repo_path)
                    )
                    
            except Exception as e:
                logger.warning(f"Error reading {file_path}: {e}")
        
        return indicators
    
    def cleanup(self, repo_path: str = None):
        """Clean up cloned repository."""
        path = repo_path or self.workspace_dir
        if os.path.exists(path):
            def on_rm_error(func, path, exc_info):
                # Handle read-only files (common with git)
                import stat
                os.chmod(path, stat.S_IWRITE)
                os.unlink(path)
                
            shutil.rmtree(path, onerror=on_rm_error)


_github_service: Optional[GitHubService] = None

def get_github_service() -> GitHubService:
    global _github_service
    if _github_service is None:
        _github_service = GitHubService()
    return _github_service

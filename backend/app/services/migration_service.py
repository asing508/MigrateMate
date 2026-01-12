"""Batch migration service for processing entire projects."""

import os
import asyncio
import zipfile
import tempfile
import shutil
import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime
import logging

from app.services.github_service import GitHubService, get_github_service
from app.services.code_parser import parse_python_file, detect_flask_routes
from app.agents.migration_agent import get_migration_agent

logger = logging.getLogger(__name__)


@dataclass
class MigrationProgress:
    """Tracks migration progress."""
    total_files: int = 0
    processed_files: int = 0
    total_chunks: int = 0
    processed_chunks: int = 0
    current_file: str = ""
    current_chunk: str = ""
    status: str = "pending"
    errors: List[str] = field(default_factory=list)
    start_time: datetime = field(default_factory=datetime.now)
    
    @property
    def percentage(self) -> float:
        if self.total_chunks == 0:
            return 0
        return (self.processed_chunks / self.total_chunks) * 100


@dataclass 
class MigrationResult:
    """Result of a file migration."""
    source_path: str
    output_path: str
    source_content: str
    migrated_content: str
    chunks_migrated: int
    chunks_failed: int
    confidence: float


class BatchMigrationService:
    """Service for migrating entire projects."""
    
    def __init__(self):
        self.github = get_github_service()
        self.agent = get_migration_agent()
        self.progress: Optional[MigrationProgress] = None
        self._progress_callback: Optional[Callable] = None
    
    def set_progress_callback(self, callback: Callable):
        """Set callback for progress updates."""
        self._progress_callback = callback
    
    def _update_progress(self, **kwargs):
        """Update progress and notify callback."""
        if self.progress:
            for key, value in kwargs.items():
                setattr(self.progress, key, value)
            if self._progress_callback:
                self._progress_callback(self.progress)
    
    async def migrate_github_repo(
        self,
        repo_url: str,
        branch: str = "main",
        source_framework: str = "flask",
        target_framework: str = "fastapi"
    ) -> Dict[str, Any]:
        """
        Migrate a GitHub repository.
        
        Returns:
            Dict with output_path, results, and summary
        """
        self.progress = MigrationProgress(status="cloning")
        self._update_progress()
        
        try:
            # Clone repository
            repo_path = self.github.clone_repo(repo_url, branch)
            self._update_progress(status="analyzing")
            
            # Detect project structure
            detection = self.github.detect_flask_project(repo_path)
            if not detection['is_flask']:
                raise ValueError("Not a Flask project - no Flask imports detected")
            
            # Find Python files
            python_files = self.github.find_python_files(repo_path)
            self._update_progress(total_files=len(python_files), status="migrating")
            
            # Create output directory
            output_dir = tempfile.mkdtemp(prefix="migratemate_output_")
            
            # Migrate each file
            results = []
            for i, file_path in enumerate(python_files):
                self._update_progress(
                    current_file=os.path.relpath(file_path, repo_path),
                    processed_files=i
                )
                
                result = await self._migrate_file(
                    file_path=file_path,
                    repo_path=repo_path,
                    output_dir=output_dir,
                    source_framework=source_framework,
                    target_framework=target_framework
                )
                results.append(result)
            
            self._update_progress(
                processed_files=len(python_files),
                status="packaging"
            )
            
            # Create output ZIP
            zip_path = self._create_output_zip(output_dir, repo_url)
            
            # Cleanup
            self.github.cleanup(repo_path)
            
            def on_rm_error(func, path, exc_info):
                # Handle read-only files (common with git)
                import stat
                os.chmod(path, stat.S_IWRITE)
                os.unlink(path)
                
            shutil.rmtree(output_dir, onerror=on_rm_error)
            
            self._update_progress(status="completed")
            
            return {
                "zip_path": zip_path,
                "results": [self._result_to_dict(r) for r in results],
                "summary": {
                    "total_files": len(python_files),
                    "files_migrated": len([r for r in results if r.chunks_migrated > 0]),
                    "total_chunks": sum(r.chunks_migrated + r.chunks_failed for r in results),
                    "chunks_succeeded": sum(r.chunks_migrated for r in results),
                    "chunks_failed": sum(r.chunks_failed for r in results),
                    "average_confidence": sum(r.confidence for r in results) / len(results) if results else 0
                }
            }
            
        except Exception as e:
            logger.error(f"Migration failed: {e}")
            self._update_progress(status="failed", errors=[str(e)])
            raise
    
    async def migrate_uploaded_zip(
        self,
        zip_path: str,
        source_framework: str = "flask",
        target_framework: str = "fastapi"
    ) -> Dict[str, Any]:
        """Migrate an uploaded ZIP file."""
        self.progress = MigrationProgress(status="extracting")
        
        # Extract ZIP
        extract_dir = tempfile.mkdtemp(prefix="migratemate_extract_")
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall(extract_dir)
        
        # Find the actual project root (might be nested)
        project_root = self._find_project_root(extract_dir)
        
        self._update_progress(status="analyzing")
        
        # Find Python files
        python_files = self.github.find_python_files(project_root)
        self._update_progress(total_files=len(python_files), status="migrating")
        
        # Create output directory
        output_dir = tempfile.mkdtemp(prefix="migratemate_output_")
        
        # Migrate each file
        results = []
        for i, file_path in enumerate(python_files):
            self._update_progress(
                current_file=os.path.relpath(file_path, project_root),
                processed_files=i
            )
            
            result = await self._migrate_file(
                file_path=file_path,
                repo_path=project_root,
                output_dir=output_dir,
                source_framework=source_framework,
                target_framework=target_framework
            )
            results.append(result)
        
        self._update_progress(processed_files=len(python_files), status="packaging")
        
        # Create output ZIP
        zip_path = self._create_output_zip(output_dir, "uploaded_project")
        
        # Cleanup
        def on_rm_error(func, path, exc_info):
            # Handle read-only files (common with git)
            import stat
            os.chmod(path, stat.S_IWRITE)
            os.unlink(path)

        shutil.rmtree(extract_dir, onerror=on_rm_error)
        shutil.rmtree(output_dir, onerror=on_rm_error)
        
        self._update_progress(status="completed")
        
        return {
            "zip_path": zip_path,
            "results": [self._result_to_dict(r) for r in results],
            "summary": {
                "total_files": len(python_files),
                "files_migrated": len([r for r in results if r.chunks_migrated > 0]),
            }
        }
    
    async def migrate_local_directory(
        self,
        source_dir: str,
        output_dir: str,
        source_framework: str = "flask",
        target_framework: str = "fastapi"
    ) -> Dict[str, Any]:
        """
        Migrate a local directory.
        
        Args:
            source_dir: Path to source Flask project
            output_dir: Path to output directory for FastAPI project
            source_framework: Source framework (flask)
            target_framework: Target framework (fastapi)
            
        Returns:
            Dict with results and summary
        """
        self.progress = MigrationProgress(status="analyzing")
        self._update_progress()
        
        try:
            # Ensure output directory exists
            os.makedirs(output_dir, exist_ok=True)
            
            # Find Python files
            python_files = self.github.find_python_files(source_dir)
            self._update_progress(total_files=len(python_files), status="migrating")
            
            # Migrate each file
            results = []
            for i, file_path in enumerate(python_files):
                self._update_progress(
                    current_file=os.path.relpath(file_path, source_dir),
                    processed_files=i
                )
                
                result = await self._migrate_file(
                    file_path=file_path,
                    repo_path=source_dir,
                    output_dir=output_dir,
                    source_framework=source_framework,
                    target_framework=target_framework
                )
                results.append(result)
            
            self._update_progress(
                processed_files=len(python_files),
                status="completed"
            )
            
            return {
                "output_dir": output_dir,
                "results": [self._result_to_dict(r) for r in results],
                "summary": {
                    "total_files": len(python_files),
                    "files_migrated": len([r for r in results if r.chunks_migrated > 0]),
                    "total_chunks": sum(r.chunks_migrated + r.chunks_failed for r in results),
                    "chunks_succeeded": sum(r.chunks_migrated for r in results),
                    "chunks_failed": sum(r.chunks_failed for r in results),
                    "average_confidence": sum(r.confidence for r in results) / len(results) if results else 0
                }
            }
            
        except Exception as e:
            logger.error(f"Migration failed: {e}")
            self._update_progress(status="failed", errors=[str(e)])
            raise
    
    async def _migrate_file(
        self,
        file_path: str,
        repo_path: str,
        output_dir: str,
        source_framework: str,
        target_framework: str
    ) -> MigrationResult:
        """Migrate a single file."""
        relative_path = os.path.relpath(file_path, repo_path)
        output_path = os.path.join(output_dir, relative_path)
        
        # Create output directory structure
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Read source
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            source_content = f.read()
        
        # Check if it's a Flask-related file that needs migration
        flask_indicators = [
            'from flask',
            'import flask',
            '@app.route',
            '@bp.route',
            '_bp.route',
            '_bp.get',
            '_bp.post',
            '_bp.put',
            '_bp.delete',
            '_bp.patch',
            'Blueprint(',
            'flask_jwt',
            'flask_caching',
            'flask_cors',
            'flask_login',
            'flask_sqlalchemy',
            'make_response(',
            'jsonify(',
            'before_request',
            'after_request',
        ]
        
        is_flask_file = any(indicator in source_content for indicator in flask_indicators)
        
        if not is_flask_file:
            # Not a Flask file, copy as-is
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(source_content)
            return MigrationResult(
                source_path=relative_path,
                output_path=relative_path,
                source_content=source_content,
                migrated_content=source_content,
                chunks_migrated=0,
                chunks_failed=0,
                confidence=1.0
            )
        
        # Parse into chunks
        chunks = parse_python_file(source_content, relative_path)
        self._update_progress(total_chunks=self.progress.total_chunks + len(chunks))
        
        # Build header
        migrated_parts = []
        chunks_migrated = 0
        chunks_failed = 0
        total_confidence = 0
        
        for chunk in chunks:
            self._update_progress(
                current_chunk=chunk.name,
                processed_chunks=self.progress.processed_chunks + 1
            )
            
            # Skip ignored chunks (whitespace)
            if chunk.chunk_type == 'ignored':
                 migrated_parts.append(chunk.content)
                 continue
            
            try:
                result = await self.agent.migrate_chunk(
                    project_id=0,
                    job_id=0,
                    source_framework=source_framework,
                    target_framework=target_framework,
                    chunk_id=0,
                    chunk_content=chunk.content,
                    chunk_name=chunk.name,
                    chunk_type=chunk.chunk_type
                )
                
                if result.get('status') == 'completed' and result.get('migrated_code'):
                    migrated_parts.append(result['migrated_code'])
                    chunks_migrated += 1
                    total_confidence += result.get('confidence_score', 0)
                else:
                    chunks_failed += 1
                    migrated_parts.append(f"# TODO: Migration failed for {chunk.name}\n{chunk.content}")
                    
            except Exception as e:
                logger.error(f"Chunk migration failed: {e}")
                chunks_failed += 1
                migrated_parts.append(f"# TODO: Migration error for {chunk.name}\n{chunk.content}")
        
        # Post-process
        migrated_content = "\n\n".join(migrated_parts)
        
        # Detect if this is the main app file (has multiple router imports)
        is_main_app = 'include_router(' in migrated_content or 'register_blueprint(' in source_content
        
        if is_main_app:
            migrated_content = self._post_process_main_app(migrated_content, source_content)
        else:
            migrated_content = self._post_process(migrated_content)
        
        # Write output
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(migrated_content)
        
        avg_confidence = total_confidence / chunks_migrated if chunks_migrated > 0 else 0
        
        return MigrationResult(
            source_path=relative_path,
            output_path=relative_path,
            source_content=source_content,
            migrated_content=migrated_content,
            chunks_migrated=chunks_migrated,
            chunks_failed=chunks_failed,
            confidence=avg_confidence
        )
    
    def _post_process_main_app(self, code: str, source_code: str) -> str:
        """
        Special post-processing for main app files (app.py).
        Handles import collisions and router naming.
        """
        # Extract original blueprint names and their associated imports from source code
        # Pattern: from routes.user_router import auth_bp, cache
        # Pattern: from routes.request_router import request_bp
        original_imports = re.findall(
            r'from\s+routes\.(\w+)\s+import\s+([^\n]+)',
            source_code
        )
        
        # Create mapping: routes module -> (blueprint_name, additional_imports)
        # e.g., {'user_router': ('auth_bp', ['cache']), 'request_router': ('request_bp', [])}
        bp_name_map = {}
        for module, imports_str in original_imports:
            imports = [i.strip() for i in imports_str.split(',')]
            bp_name = None
            additional = []
            for imp in imports:
                if imp.endswith('_bp'):
                    bp_name = imp
                else:
                    additional.append(imp)
            if bp_name:
                bp_name_map[module] = (bp_name, additional)
        
        # Find all router imports in migrated code that need fixing
        # Pattern: from routes.user_router import router
        router_import_pattern = r'from\s+routes\.(\w+)\s+import\s+router([^\w]|$)'
        router_modules = re.findall(router_import_pattern, code)
        
        if router_modules:
            # We have collision - multiple 'router' imports
            for module, _ in router_modules:
                if module in bp_name_map:
                    bp_name, additional = bp_name_map[module]
                    
                    # Build the correct import statement
                    all_imports = [bp_name] + additional
                    new_import = f"from routes.{module} import {', '.join(all_imports)}"
                    
                    # Fix the import statement - handle various patterns
                    # Pattern 1: from routes.X import router (end of line)
                    code = re.sub(
                        rf'from\s+routes\.{module}\s+import\s+router\s*$',
                        new_import,
                        code,
                        flags=re.MULTILINE
                    )
                    # Pattern 2: from routes.X import router, something
                    code = re.sub(
                        rf'from\s+routes\.{module}\s+import\s+router\s*,',
                        f'from routes.{module} import {bp_name},',
                        code
                    )
                    # Pattern 3: from routes.X import router (followed by newline or something)
                    code = re.sub(
                        rf'from\s+routes\.{module}\s+import\s+router(\s)',
                        rf'from routes.{module} import {bp_name}\1',
                        code
                    )
                else:
                    # No mapping found, create a reasonable name
                    bp_name = f"{module.replace('_router', '')}_bp"
                    code = re.sub(
                        rf'from\s+routes\.{module}\s+import\s+router(\s|$)',
                        rf'from routes.{module} import {bp_name}\1',
                        code,
                        flags=re.MULTILINE
                    )
        
        # Now fix the include_router calls
        # Find the original order from source code
        original_bp_order = re.findall(
            r'app\.register_blueprint\((\w+)\)',
            source_code
        )
        
        # Replace generic include_router(router) calls with correct names in order
        if original_bp_order:
            for original_bp in original_bp_order:
                # Replace one occurrence at a time
                code = re.sub(
                    r'app\.include_router\(router\)',
                    f'app.include_router({original_bp})',
                    code,
                    count=1
                )
        
        # Apply standard post-processing
        code = self._post_process(code)
        
        return code
    
    def _generate_fastapi_header(self) -> str:
        """Generate FastAPI file header."""
        return '''"""Auto-migrated from Flask to FastAPI by MigrateMate."""

from fastapi import FastAPI, HTTPException, Request, Depends
from pydantic import BaseModel
from typing import List, Optional

app = FastAPI(title="Migrated API")
'''
    
    def _post_process(self, code: str) -> str:
        """Post-process migrated code to clean up and fix common issues."""
        
        # 1. Fix Flask API references
        code = code.replace('"Welcome to Flask', '"Welcome to FastAPI')
        code = code.replace("'Welcome to Flask", "'Welcome to FastAPI")
        
        # 2. Fix any remaining Flask patterns that the agent might have missed
        # Blueprint to APIRouter - but PRESERVE the variable name!
        code = re.sub(
            r"(\w+)\s*=\s*Blueprint\s*\(\s*['\"](\w+)['\"],\s*__name__\s*\)",
            r"\1 = APIRouter(tags=['\2'])",
            code
        )
        
        # 3. Fix register_blueprint to include_router
        code = code.replace('.register_blueprint(', '.include_router(')

        # 4. FIX: Cleanup request.endpoint if patterns missed it
        if 'request.endpoint' in code:
             # Common pattern: if request.endpoint == 'endpoint_name':
             code = code.replace('request.endpoint', 'str(request.url.path)')
             # Add warning since FastAPI paths might differ from Flask endpoints
             code = re.sub(
                 r'if\s+str\(request\.url\.path\)\s*==', 
                 '# TODO: Verify path (request.endpoint is not available in FastAPI)\n        if str(request.url.path) ==', 
                 code
             )

        # 5. Add import json if json.loads/json.dumps is used but not imported
        if ('json.loads' in code or 'json.dumps' in code) and 'import json' not in code:
            lines = code.split('\n')
            insert_idx = 0
            for i, line in enumerate(lines):
                if line.startswith('from ') or line.startswith('import '):
                    insert_idx = i + 1
            if insert_idx > 0:
                lines.insert(insert_idx, 'import json')
                code = '\n'.join(lines)
            else:
                code = 'import json\n' + code
        
        # 6. Fix any remaining Flask imports that slipped through
        flask_import_fixes = [
            ('from flask import Flask', 'from fastapi import FastAPI'),
            ('from flask import Blueprint', 'from fastapi import APIRouter'),
            ('from flask import request, jsonify', 'from fastapi import Request'),
            ('from flask import request', 'from fastapi import Request'),
            ('from flask import jsonify', ''),
            ('from flask import make_response', 'from fastapi.responses import JSONResponse'),
            ('from flask import json', 'import json'),  # Convert Flask json to standard json
        ]
        for old, new in flask_import_fixes:
            if old in code and new:
                code = code.replace(old, new)
            elif old in code and not new:
                code = code.replace(old, '# Removed: ' + old)

        # 7. FIX: ConnectionManager Deduplication
        # If chunks generated multiple ConnectionManager classes, keep only the first one
        if code.count("class ConnectionManager:") > 1:
            # Protect the first instance with a unique marker
            code = code.replace("class ConnectionManager:", "__KEEP_THIS_CM__", 1)
            
            # Remove subsequent instances and their instantiations
            # Regex removes "class ConnectionManager:... manager = ConnectionManager()"
            code = re.sub(
                r'class ConnectionManager:[\s\S]+?manager = ConnectionManager\(\)', 
                '', 
                code
            )
            # Remove any stragglers
            code = code.replace("class ConnectionManager:", "")
            
            # Restore the first instance
            code = code.replace("__KEEP_THIS_CM__", "class ConnectionManager:")

        # 8. FIX: Comment out SocketIO decorators that might cause errors
        code = re.sub(r'^(?!.*#)(.*)@socketio\.on', r'# TODO: \1@socketio.on', code, flags=re.MULTILINE)
        
        # 9. Ensure newlines before decorators
        code = re.sub(r'(\S)(@app\.)', r'\1\n\n\2', code)
        code = re.sub(r'(\S)(@router\.)', r'\1\n\n\2', code)
        code = re.sub(r'(\S)(@\w+_bp\.)', r'\1\n\n\2', code)
        
        # 10. Fix double async
        code = code.replace('async async def', 'async def')
        
        # 11. Remove excessive blank lines (more than 2)
        code = re.sub(r'\n{4,}', '\n\n\n', code)
        
        # 12. Remove duplicate imports
        lines = code.split('\n')
        seen_imports = set()
        result = []
        for line in lines:
            stripped = line.strip()
            if stripped.startswith('from ') or stripped.startswith('import '):
                normalized = stripped.replace('  ', ' ')
                if normalized not in seen_imports:
                    seen_imports.add(normalized)
                    result.append(line)
            else:
                result.append(line)
        
        # Remove empty # Removed: lines
        result = [line for line in result if line.strip() != '# Removed:']
        
        code = '\n'.join(result)
        
        # 13. Ensure imports are at the top of the file
        code = self._organize_imports(code)
        
        return code
    
    def _organize_imports(self, code: str) -> str:
        """Organize imports to be at the top of the file."""
        lines = code.split('\n')
        
        import_lines = []
        other_lines = []
        docstring_lines = []
        comment_lines = []  # TODO comments at the top
        in_docstring = False
        docstring_done = False
        
        for i, line in enumerate(lines):
            stripped = line.strip()
            
            # Handle module docstrings at the start
            if not docstring_done:
                if stripped.startswith('"""') or stripped.startswith("'''"):
                    in_docstring = not in_docstring or stripped.count('"""') >= 2 or stripped.count("'''") >= 2
                    docstring_lines.append(line)
                    if not in_docstring:
                        docstring_done = True
                    continue
                elif in_docstring:
                    docstring_lines.append(line)
                    if '"""' in stripped or "'''" in stripped:
                        in_docstring = False
                        docstring_done = True
                    continue
                else:
                    docstring_done = True
            
            # Categorize lines
            if stripped.startswith('from ') or stripped.startswith('import '):
                import_lines.append(line)
            elif stripped.startswith('# TODO:') and not other_lines:
                # Keep TODO comments near the top if they're before any code
                comment_lines.append(line)
            else:
                other_lines.append(line)
        
        # Rebuild the file with proper organization
        result_parts = []
        if docstring_lines:
            result_parts.extend(docstring_lines)
            result_parts.append('')
        if comment_lines:
            result_parts.extend(comment_lines)
        if import_lines:
            result_parts.extend(import_lines)
            result_parts.append('')
            result_parts.append('')
        result_parts.extend(other_lines)
        
        # Clean up leading/trailing whitespace
        return '\n'.join(result_parts).strip() + '\n'
    
    def _find_project_root(self, extract_dir: str) -> str:
        """Find the actual project root in extracted ZIP."""
        # Check if there's a single directory at root
        items = os.listdir(extract_dir)
        if len(items) == 1:
            single_item = os.path.join(extract_dir, items[0])
            if os.path.isdir(single_item):
                return single_item
        return extract_dir
    
    def _create_output_zip(self, output_dir: str, project_name: str) -> str:
        """Create output ZIP file."""
        # Clean project name
        project_name = project_name.split('/')[-1].replace('.git', '')
        
        zip_path = os.path.join(
            tempfile.gettempdir(),
            f"{project_name}_fastapi_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip"
        )
        
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
            for root, dirs, files in os.walk(output_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, output_dir)
                    zf.write(file_path, arcname)
            
            # Add requirements.txt with common FastAPI dependencies
            requirements = """# FastAPI dependencies (auto-generated by MigrateMate)
fastapi==0.111.0
uvicorn[standard]==0.30.1
pydantic==2.7.4
python-multipart==0.0.9  # For form data
python-jose[cryptography]==3.3.0  # For JWT authentication
passlib[bcrypt]==1.7.4  # For password hashing
starlette==0.37.2  # ASGI framework
httpx==0.27.0  # Async HTTP client
"""
            zf.writestr("requirements.txt", requirements)
        
        return zip_path
    
    def _result_to_dict(self, result: MigrationResult) -> Dict[str, Any]:
        """Convert MigrationResult to dict."""
        return {
            "source_path": result.source_path,
            "output_path": result.output_path,
            "source_content": result.source_content,
            "migrated_content": result.migrated_content,
            "chunks_migrated": result.chunks_migrated,
            "chunks_failed": result.chunks_failed,
            "confidence": result.confidence
        }


_batch_service: Optional[BatchMigrationService] = None

def get_batch_migration_service() -> BatchMigrationService:
    global _batch_service
    if _batch_service is None:
        _batch_service = BatchMigrationService()
    return _batch_service

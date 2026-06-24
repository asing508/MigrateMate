"""Batch migration service for processing entire projects."""

import os
import zipfile
import tempfile
import shutil
import re
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime
import logging

from app.services.github_service import get_github_service
from app.services.code_parser import parse_python_file
from app.agents.migration_agent import get_migration_agent
# MigrationProgress now lives in job_store so a single migration owns its own
# state (see job_store.py for why). Re-exported here for backwards-compatibility.
from app.services.job_store import MigrationProgress

logger = logging.getLogger(__name__)


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
    """Service for migrating entire projects.

    The service is *stateless* with respect to progress: each migration carries
    its own :class:`MigrationProgress`, passed in by the caller. That makes
    concurrent migrations correct (no shared mutable state) and keeps the
    service safe to use as a singleton.
    """

    def __init__(self):
        self.github = get_github_service()
        self.agent = get_migration_agent()

    @staticmethod
    def _detached_progress(kind: str = "local") -> MigrationProgress:
        """A throwaway progress object for direct/programmatic calls."""
        return MigrationProgress(migration_id=f"{kind}-direct", kind=kind)

    @staticmethod
    def _summarize(results: List["MigrationResult"]) -> Dict[str, Any]:
        """Build the summary block. Average confidence is computed over files
        that actually had chunks migrated, so copied (non-Flask) files no longer
        inflate the score."""
        migrated = [r for r in results if r.chunks_migrated > 0]
        return {
            "total_files": len(results),
            "files_migrated": len(migrated),
            "total_chunks": sum(r.chunks_migrated + r.chunks_failed for r in results),
            "chunks_succeeded": sum(r.chunks_migrated for r in results),
            "chunks_failed": sum(r.chunks_failed for r in results),
            "average_confidence": (
                sum(r.confidence for r in migrated) / len(migrated) if migrated else 0.0
            ),
        }

    @staticmethod
    def _rmtree_quiet(path: str) -> None:
        """Remove a tree, tolerating read-only files (common with git on Windows)."""
        def on_rm_error(func, p, exc_info):
            import stat
            try:
                os.chmod(p, stat.S_IWRITE)
                os.unlink(p)
            except OSError:
                pass
        shutil.rmtree(path, onerror=on_rm_error)

    async def migrate_github_repo(
        self,
        repo_url: str,
        branch: str = "main",
        source_framework: str = "flask",
        target_framework: str = "fastapi",
        progress: Optional[MigrationProgress] = None,
    ) -> Dict[str, Any]:
        """
        Migrate a GitHub repository.

        Returns:
            Dict with zip_path, results, and summary
        """
        progress = progress or self._detached_progress("github")
        progress.status = "running"
        progress.start_step("fetch", label="Cloning repository", detail=repo_url)

        output_dir = None
        repo_path = None
        try:
            # Clone repository
            repo_path = self.github.clone_repo(repo_url, branch)

            progress.start_step("analyze", detail="Detecting Flask project")
            detection = self.github.detect_flask_project(repo_path)
            if not detection['is_flask']:
                raise ValueError("Not a Flask project - no Flask imports detected")

            python_files = self.github.find_python_files(repo_path)
            progress.total_files = len(python_files)
            progress.start_step("migrate", detail=f"{len(python_files)} files")

            output_dir = tempfile.mkdtemp(prefix="migratemate_output_")

            results = []
            for i, file_path in enumerate(python_files):
                progress.current_file = os.path.relpath(file_path, repo_path)
                progress.touch()
                result = await self._migrate_file(
                    file_path=file_path,
                    repo_path=repo_path,
                    output_dir=output_dir,
                    source_framework=source_framework,
                    target_framework=target_framework,
                    progress=progress,
                )
                results.append(result)
                progress.processed_files = i + 1
                progress.touch()

            progress.start_step("package", detail="Building ZIP")
            zip_path = self._create_output_zip(output_dir, repo_url)

            progress.finish()
            summary = self._summarize(results)
            return {
                "zip_path": zip_path,
                "results": [self._result_to_dict(r) for r in results],
                "summary": summary,
            }

        except Exception as e:
            logger.error(f"Migration failed: {e}")
            progress.fail_current(str(e))
            raise
        finally:
            if repo_path:
                try:
                    self.github.cleanup(repo_path)
                except OSError:
                    pass
            if output_dir and os.path.exists(output_dir):
                self._rmtree_quiet(output_dir)
    
    async def migrate_uploaded_zip(
        self,
        zip_path: str,
        source_framework: str = "flask",
        target_framework: str = "fastapi",
        progress: Optional[MigrationProgress] = None,
    ) -> Dict[str, Any]:
        """Migrate an uploaded ZIP file."""
        progress = progress or self._detached_progress("upload")
        progress.status = "running"
        progress.start_step("fetch", label="Extracting archive")

        extract_dir = tempfile.mkdtemp(prefix="migratemate_extract_")
        output_dir = None
        try:
            with zipfile.ZipFile(zip_path, 'r') as zf:
                self._safe_extract(zf, extract_dir)

            project_root = self._find_project_root(extract_dir)

            progress.start_step("analyze")
            python_files = self.github.find_python_files(project_root)
            progress.total_files = len(python_files)
            progress.start_step("migrate", detail=f"{len(python_files)} files")

            output_dir = tempfile.mkdtemp(prefix="migratemate_output_")

            results = []
            for i, file_path in enumerate(python_files):
                progress.current_file = os.path.relpath(file_path, project_root)
                progress.touch()
                result = await self._migrate_file(
                    file_path=file_path,
                    repo_path=project_root,
                    output_dir=output_dir,
                    source_framework=source_framework,
                    target_framework=target_framework,
                    progress=progress,
                )
                results.append(result)
                progress.processed_files = i + 1
                progress.touch()

            progress.start_step("package", detail="Building ZIP")
            out_zip = self._create_output_zip(output_dir, "uploaded_project")

            progress.finish()
            return {
                "zip_path": out_zip,
                "results": [self._result_to_dict(r) for r in results],
                "summary": self._summarize(results),
            }
        except Exception as e:
            logger.error(f"Migration failed: {e}")
            progress.fail_current(str(e))
            raise
        finally:
            self._rmtree_quiet(extract_dir)
            if output_dir and os.path.exists(output_dir):
                self._rmtree_quiet(output_dir)

    async def migrate_local_directory(
        self,
        source_dir: str,
        output_dir: str,
        source_framework: str = "flask",
        target_framework: str = "fastapi",
        progress: Optional[MigrationProgress] = None,
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
        progress = progress or self._detached_progress("local")
        progress.status = "running"
        progress.start_step("analyze", detail=source_dir)

        try:
            os.makedirs(output_dir, exist_ok=True)

            python_files = self.github.find_python_files(source_dir)
            progress.total_files = len(python_files)
            progress.start_step("migrate", detail=f"{len(python_files)} files")

            results = []
            for i, file_path in enumerate(python_files):
                progress.current_file = os.path.relpath(file_path, source_dir)
                progress.touch()
                result = await self._migrate_file(
                    file_path=file_path,
                    repo_path=source_dir,
                    output_dir=output_dir,
                    source_framework=source_framework,
                    target_framework=target_framework,
                    progress=progress,
                )
                results.append(result)
                progress.processed_files = i + 1
                progress.touch()

            progress.finish()
            return {
                "output_dir": output_dir,
                "results": [self._result_to_dict(r) for r in results],
                "summary": self._summarize(results),
            }

        except Exception as e:
            logger.error(f"Migration failed: {e}")
            progress.fail_current(str(e))
            raise
    
    async def _migrate_file(
        self,
        file_path: str,
        repo_path: str,
        output_dir: str,
        source_framework: str,
        target_framework: str,
        progress: Optional[MigrationProgress] = None,
    ) -> MigrationResult:
        """Migrate a single file."""
        progress = progress or self._detached_progress()
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
        progress.total_chunks += len(chunks)
        progress.touch()

        # Build header
        migrated_parts = []
        chunks_migrated = 0
        chunks_failed = 0
        total_confidence = 0

        for chunk in chunks:
            progress.current_chunk = chunk.name
            progress.processed_chunks += 1
            progress.touch()

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
        """Special post-processing for main app files (app.py).

        Handles import collisions when multiple blueprint modules each get
        re-exported as `router` after migration, and rewrites
        `app.register_blueprint(<bp>)` to `app.include_router(<bp>)` using the
        ORIGINAL blueprint variable names from the Flask source.

        Generalised: works for any `from <pkg.path> import <name>` (not just
        `from routes.<X>`), and accepts blueprint variables that don't end in
        `_bp` by matching against the actual Blueprint(...) declarations in the
        source.
        """
        # 1) Authoritative blueprint names come from BOTH places they appear in
        #    the original source: `Blueprint(...)` declarations (when they live
        #    in this file) AND `register_blueprint(<name>)` calls (when they
        #    were imported from submodules — the typical multi-file layout).
        bp_vars_in_source = set(
            re.findall(r'(\w+)\s*=\s*Blueprint\s*\(', source_code)
        )
        bp_vars_in_source |= set(
            re.findall(r'\.register_blueprint\(\s*(\w+)', source_code)
        )

        # 2) Collect every `from X import ...` line in the source so we can map
        #    "module path -> (blueprint_name, [other_imports])".
        original_imports = re.findall(
            r'from\s+([\w\.]+)\s+import\s+([^\n]+)',
            source_code,
        )

        bp_name_map: dict[str, tuple[str, list[str]]] = {}
        for module, imports_str in original_imports:
            imports = [i.strip() for i in imports_str.split(',') if i.strip()]
            bp_name = None
            additional = []
            for imp in imports:
                # Prefer matching against the authoritative bp name set; fall
                # back to the `_bp`/`_blueprint` suffix heuristic only if we
                # have nothing better.
                is_bp = (
                    imp in bp_vars_in_source
                    or (not bp_vars_in_source and (imp.endswith('_bp') or imp.endswith('_blueprint')))
                )
                if is_bp and bp_name is None:
                    bp_name = imp
                else:
                    additional.append(imp)
            if bp_name:
                bp_name_map[module] = (bp_name, additional)

        # 3) Find every `from <module> import router[, ...]` collision in the
        #    migrated code and rewrite using the original bp name.
        router_modules = set(
            re.findall(r'from\s+([\w\.]+)\s+import\s+router(?:[^\w]|$)', code)
        )

        for module in router_modules:
            if module in bp_name_map:
                bp_name, additional = bp_name_map[module]
                all_imports = [bp_name] + additional
                new_import = f"from {module} import {', '.join(all_imports)}"

                # Replace the entire `from <module> import ...router...` line.
                code = re.sub(
                    rf'^from\s+{re.escape(module)}\s+import\s+[^\n]*\brouter\b[^\n]*$',
                    new_import,
                    code,
                    flags=re.MULTILINE,
                )
            else:
                # Synthesize a reasonable name from the trailing module segment.
                last = module.rsplit('.', 1)[-1]
                bp_name = f"{last.replace('_router', '').replace('_routes', '')}_bp"
                code = re.sub(
                    rf'(^from\s+{re.escape(module)}\s+import\s+)router\b',
                    rf'\1{bp_name}',
                    code,
                    flags=re.MULTILINE,
                )

        # 4) Fix `app.include_router(router)` calls. Use the ORIGINAL
        #    register_blueprint(...) order from the source for correctness.
        original_bp_order = re.findall(
            r'\.register_blueprint\(\s*(\w+)',
            source_code,
        )
        for original_bp in original_bp_order:
            code = re.sub(
                r'\.include_router\(\s*router\s*([,\)])',
                lambda m, name=original_bp: f'.include_router({name}{m.group(1)}',
                code,
                count=1,
            )

        return self._post_process(code)
    
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
        # Blueprint to APIRouter - but PRESERVE the variable name! Allow extra
        # kwargs like url_prefix='/auth', template_folder='...', static_folder='...'.
        def _bp_to_router(m: 're.Match') -> str:
            var = m.group(1)
            tag = m.group(2)
            extra = m.group(3) or ''
            url_prefix_match = re.search(r"url_prefix\s*=\s*['\"]([^'\"]+)['\"]", extra)
            kwargs = [f"tags=['{tag}']"]
            if url_prefix_match:
                kwargs.append(f"prefix='{url_prefix_match.group(1)}'")
            return f"{var} = APIRouter({', '.join(kwargs)})"

        code = re.sub(
            r"(\w+)\s*=\s*Blueprint\s*\(\s*['\"]([\w\.]+)['\"]\s*,\s*__name__([^)]*)\)",
            _bp_to_router,
            code,
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
        
        # 12. Remove duplicate TOP-LEVEL imports (preserve indented function-local imports)
        lines = code.split('\n')
        seen_imports = set()
        result = []
        for line in lines:
            stripped = line.strip()
            is_top_level_import = (
                (line.startswith('from ') or line.startswith('import '))
                and (stripped.startswith('from ') or stripped.startswith('import '))
            )
            if is_top_level_import:
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
        """Organize TOP-LEVEL imports to be at the top of the file.

        Only unindented imports are hoisted; indented imports inside functions
        (used for lazy loading or to avoid circular imports) stay in place.
        """
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

            # Only hoist UNINDENTED imports — leaving function-local imports
            # (which are intentionally lazy / break circular deps) where they are.
            is_top_level_import = (
                (line.startswith('from ') or line.startswith('import '))
                and (stripped.startswith('from ') or stripped.startswith('import '))
            )

            if is_top_level_import:
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
    
    @staticmethod
    def _safe_extract(zf: zipfile.ZipFile, dest_dir: str) -> None:
        """Extract a ZIP, refusing any member that would escape ``dest_dir``.

        Guards against "zip slip" — archive entries like ``../../etc/passwd`` or
        absolute paths that would otherwise let an uploaded ZIP overwrite files
        outside the extraction directory.
        """
        dest_root = os.path.realpath(dest_dir)
        for member in zf.namelist():
            target = os.path.realpath(os.path.join(dest_dir, member))
            if target != dest_root and not target.startswith(dest_root + os.sep):
                raise ValueError(f"Unsafe path in archive (zip slip blocked): {member!r}")
        zf.extractall(dest_dir)

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

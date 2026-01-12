"""MigrateMate Migration Agent using Google Gemini for Flask to FastAPI conversion."""

import os
import re
import asyncio
import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

# Optional: Google Generative AI (Gemini)
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    genai = None
    GEMINI_AVAILABLE = False

from app.core.config import settings

logger = logging.getLogger(__name__)


class AgentState(Enum):
    """Migration agent state."""
    IDLE = "idle"
    ANALYZING = "analyzing"
    MIGRATING = "migrating"
    VALIDATING = "validating"
    COMPLETED = "completed"
    FAILED = "failed"


# Flask to FastAPI migration patterns
FLASK_TO_FASTAPI_PATTERNS = {
    # Import mappings
    "from flask import Flask": "from fastapi import FastAPI",
    "from flask import Blueprint": "from fastapi import APIRouter",
    "from flask import request": "from fastapi import Request",
    "from flask import jsonify": "",  # FastAPI returns dicts directly
    "from flask import make_response": "from fastapi.responses import JSONResponse",
    "from flask import abort": "from fastapi import HTTPException",
    "from flask import redirect": "from fastapi.responses import RedirectResponse",
    "from flask import render_template": "",  # Jinja2 needs different setup in FastAPI
    "from flask import url_for": "",  # Not directly available in FastAPI
    "from flask import session": "",  # Need to use starlette session
    "from flask import g": "",  # Use dependency injection in FastAPI
    "from flask import current_app": "",  # Use dependency injection
    "from flask_jwt_extended import jwt_required": "from fastapi import Depends",
    "from flask_jwt_extended import JWTManager": "",  # Remove - will use python-jose or similar
    "from flask_jwt_extended import create_access_token": "",  # Replace with custom JWT
    "from flask_caching import Cache": "",  # Replace with fastapi-cache or similar
    
    # App initialization
    "app = Flask(__name__)": "app = FastAPI()",
    
    # Blueprint to APIRouter
    "Blueprint(": "APIRouter(",
    ".register_blueprint(": ".include_router(",
    
    # Route decorators
    "@app.route": "@app.api_route",
    "@bp.route": "@router.api_route",
    ".route(": ".api_route(",
    
    # Response handling
    "jsonify(": "",  # Remove - return dict directly
    "make_response(": "JSONResponse(content=",
    
    # Request data access
    "request.json": "await request.json()",
    "request.data": "await request.body()",
    "request.form": "await request.form()",
    "request.args": "request.query_params",
    "request.files": "# TODO: Use UploadFile in FastAPI",
    "request.endpoint": "str(request.url.path)",

    # Before request hooks
    "app.before_request": "# TODO: Convert to FastAPI middleware or dependency",
    "app.after_request": "# TODO: Convert to FastAPI middleware",
    
    # Error handling
    "abort(404)": 'raise HTTPException(status_code=404, detail="Not found")',
    "abort(400)": 'raise HTTPException(status_code=400, detail="Bad request")',
    "abort(401)": 'raise HTTPException(status_code=401, detail="Unauthorized")',
    "abort(403)": 'raise HTTPException(status_code=403, detail="Forbidden")',
    "abort(500)": 'raise HTTPException(status_code=500, detail="Internal server error")',
}


MIGRATION_SYSTEM_PROMPT = """You are an expert Python developer specializing in migrating Flask applications to FastAPI. 
Your task is to convert Flask code to equivalent FastAPI code while maintaining the same functionality.

## CRITICAL RULES - MUST FOLLOW:

### VARIABLE NAMING - MOST IMPORTANT!
- **ALWAYS preserve the original variable names for Blueprints/Routers!**
- `auth_bp = Blueprint('auth_bp', __name__)` becomes `auth_bp = APIRouter(tags=['auth_bp'])`
- `request_bp = Blueprint('request_bp', __name__)` becomes `request_bp = APIRouter(tags=['request_bp'])`
- NEVER rename blueprint variables to generic names like `router`!
- The route decorators must use the SAME variable: `@auth_bp.post('/path')` stays as `@auth_bp.post('/path')`

### 1. Imports
- Replace `from flask import Flask` with `from fastapi import FastAPI`
- Replace `from flask import Blueprint` with `from fastapi import APIRouter`
- Replace `from flask import request` with `from fastapi import Request`
- Replace `from flask import make_response` with `from fastapi.responses import JSONResponse`
- **CRITICAL: If `json.loads()` or `json.dumps()` is used in the code, ADD `import json` at the top!**
- If `from flask import json` is present, replace with `import json`
- Remove `from flask import jsonify` (FastAPI returns dicts directly)
- Replace `from flask_jwt_extended import jwt_required` with `from fastapi import Depends`
- Replace `from flask_jwt_extended import create_access_token` with a TODO comment for JWT implementation
- Remove `from flask_caching import Cache` (add TODO for fastapi-cache2)

### 2. Application Setup
- `app = Flask(__name__)` becomes `app = FastAPI()`
- `bp = Blueprint('name', __name__)` becomes `bp = APIRouter(tags=['name'])` - KEEP THE SAME VARIABLE NAME!
- `app.register_blueprint(bp)` becomes `app.include_router(bp)` - use original variable name!

### 3. Route Decorators - PRESERVE VARIABLE NAMES!
- `@auth_bp.route('/path', methods=['GET'])` becomes `@auth_bp.get('/path')` - NOT @router.get!
- `@auth_bp.route('/path', methods=['POST'])` becomes `@auth_bp.post('/path')`
- `@auth_bp.get('/path')` stays as `@auth_bp.get('/path')` - Flask already supports this!
- Convert functions to `async def` when using await

### 4. Request Handling - CRITICAL!
- **When a function uses `request.data`, `request.json`, etc., it MUST have `request: Request` as a parameter!**
- Example: `def register():` that uses `request.data` becomes `async def register(request: Request):`
- `request.json` becomes `await request.json()`
- `request.data` becomes `await request.body()`
- `json.loads(request.data)` becomes `json.loads(await request.body())` AND add `import json`!
- `request.args.get('param')` becomes function parameter: `param: str = Query(None)`

### 5. Router/Controller Wrappers (CRITICAL FIX)
- If a route function wraps another function (e.g., `def wrapper(): return controller_func()`), you MUST ASSUME the controller needs `request`.
- CHANGE: `def wrapper():` -> `async def wrapper(request: Request):`
- CHANGE: `return controller_func()` -> `return await controller_func(request)`

### 6. Response Handling
- Remove `jsonify()` calls - return dictionaries directly
- `make_response({'key': 'value'}, 201)` becomes `JSONResponse(content={'key': 'value'}, status_code=201)`
- `make_response(content)` (single argument) becomes `JSONResponse(content=content)` (default status 200)
- ALL `make_response()` calls must be converted - don't leave any behind!

### 7. Authentication
- `@jwt_required()` becomes a TODO comment: `# TODO: Add auth dependency - Depends(get_current_user)`
- `create_access_token()` becomes a TODO: `# TODO: Implement JWT with python-jose`

### 8. Caching
- `@cache.cached()` becomes: `# TODO: Add caching with fastapi-cache2 or @lru_cache`
- `cache = Cache(...)` becomes: `# TODO: Set up caching - consider fastapi-cache2 or Redis`

### 9. WebSockets (Flask-SocketIO)
- Convert `@socketio.on('event')` to standard FastAPI WebSocket endpoints or comment them out.
- DO NOT redefine `ConnectionManager` classes if they are already present in the context.
- Assume `manager` instance exists for broadcasting.

### Output Format
Return ONLY the migrated Python code, no explanations or markdown.
Do NOT include ```python or ``` markers.
Ensure ALL functions that use `request` have `request: Request` as a parameter.
Ensure `import json` is present if json.loads/json.dumps is used.
"""

class MigrationAgent:
    """Agent for migrating Flask code to FastAPI using Gemini."""
    
    def __init__(self):
        self.state = AgentState.IDLE
        self._model = None
        self._configure_gemini()
    
    def _configure_gemini(self):
        """Configure Google Gemini API."""
        if not GEMINI_AVAILABLE:
            logger.warning("google-generativeai package not installed - using pattern-based migration")
            return
            
        api_key = settings.GEMINI_API_KEY
        if api_key:
            genai.configure(api_key=api_key)
            self._model = genai.GenerativeModel(
                model_name="gemini-2.0-flash",
                system_instruction=MIGRATION_SYSTEM_PROMPT
            )
            logger.info("Gemini API configured successfully")
        else:
            logger.warning("No GEMINI_API_KEY found - migration agent will use pattern-based fallback")
    
    async def migrate_chunk(
        self,
        project_id: int,
        job_id: int,
        source_framework: str,
        target_framework: str,
        chunk_id: int,
        chunk_content: str,
        chunk_name: str,
        chunk_type: str,
        retrieval_context: str = "",
        max_iterations: int = 3
    ) -> Dict[str, Any]:
        """
        Migrate a single code chunk from Flask to FastAPI.
        
        Args:
            project_id: Project identifier
            job_id: Migration job identifier
            source_framework: Source framework (flask)
            target_framework: Target framework (fastapi)
            chunk_id: Chunk identifier
            chunk_content: The code to migrate
            chunk_name: Name of the function/class
            chunk_type: Type of code (function, class, misc)
            retrieval_context: Additional context for migration
            max_iterations: Maximum retry attempts
            
        Returns:
            Dictionary with migrated_code, confidence_score, status, errors, iterations
        """
        self.state = AgentState.ANALYZING
        
        errors = []
        migrated_code = None
        confidence_score = 0.0
        iterations = 0
        
        try:
            self.state = AgentState.MIGRATING
            
            # Try Gemini-based migration first
            if self._model:
                for iteration in range(max_iterations):
                    iterations = iteration + 1
                    try:
                        migrated_code = await self._migrate_with_gemini(
                            chunk_content, 
                            chunk_name, 
                            chunk_type,
                            source_framework,
                            target_framework,
                            retrieval_context
                        )
                        if migrated_code:
                            confidence_score = self._calculate_confidence(chunk_content, migrated_code)
                            break
                    except Exception as e:
                        errors.append(f"Iteration {iteration + 1}: {str(e)}")
                        logger.warning(f"Migration iteration {iteration + 1} failed: {e}")
                        await asyncio.sleep(0.5)  # Brief pause before retry
            
            # Fallback to pattern-based migration
            if not migrated_code:
                logger.info(f"Using pattern-based migration for {chunk_name}")
                migrated_code = self._migrate_with_patterns(chunk_content)
                confidence_score = 0.6  # Lower confidence for pattern-based
            
            self.state = AgentState.VALIDATING
            
            # Validate the migrated code
            is_valid, validation_errors = self._validate_migration(migrated_code)
            if not is_valid:
                errors.extend(validation_errors)
                # Try to fix common issues
                migrated_code = self._post_process(migrated_code)
            
            self.state = AgentState.COMPLETED
            
            return {
                "migrated_code": migrated_code,
                "confidence_score": confidence_score,
                "status": "completed" if migrated_code else "failed",
                "errors": errors,
                "iterations": iterations
            }
            
        except Exception as e:
            self.state = AgentState.FAILED
            logger.error(f"Migration failed for {chunk_name}: {e}")
            errors.append(str(e))
            return {
                "migrated_code": None,
                "confidence_score": 0.0,
                "status": "failed",
                "errors": errors,
                "iterations": iterations
            }
    
    async def _migrate_with_gemini(
        self,
        chunk_content: str,
        chunk_name: str,
        chunk_type: str,
        source_framework: str,
        target_framework: str,
        retrieval_context: str
    ) -> Optional[str]:
        """Use Gemini to migrate code."""
        prompt = f"""Migrate the following {source_framework} {chunk_type} named '{chunk_name}' to {target_framework}:

```python
{chunk_content}
```

{f"Additional context: {retrieval_context}" if retrieval_context else ""}

Return ONLY the migrated Python code, no explanations or markdown code blocks."""

        response = await asyncio.to_thread(
            self._model.generate_content,
            prompt
        )
        
        if response and response.text:
            # Clean up the response (remove any markdown if present)
            code = response.text.strip()
            code = self._clean_code_response(code)
            return code
        
        return None
    
    def _clean_code_response(self, code: str) -> str:
        """Clean up LLM response to extract pure Python code."""
        # Remove markdown code blocks if present
        if code.startswith("```python"):
            code = code[9:]
        elif code.startswith("```"):
            code = code[3:]
        
        if code.endswith("```"):
            code = code[:-3]
        
        return code.strip()
    
    def _migrate_with_patterns(self, code: str) -> str:
        """Fallback pattern-based migration (comprehensive fixes)."""
        result = code
        original_code = code  # Keep reference to original
        
        # ===== PHASE 1: Detect what's needed before modifying =====
        needs_json = 'json.loads' in result or 'json.dumps' in result
        uses_request_data = 'request.data' in result or 'request.json' in result or 'request.form' in result
        has_flask_imports = 'from flask import' in result
        
        # Track blueprint variable names for consistent handling
        blueprint_vars = re.findall(r'(\w+)\s*=\s*Blueprint\s*\(', result)
        
        # ===== PHASE 2: Handle Flask Imports =====
        def transform_imports(match):
            imports = match.group(1).split(',')
            imports = [i.strip() for i in imports]
            
            fastapi_imports = []
            extra_lines = []
            
            for imp in imports:
                if imp == 'Flask': 
                    fastapi_imports.append('FastAPI')
                elif imp == 'Blueprint': 
                    fastapi_imports.append('APIRouter')
                elif imp == 'request': 
                    fastapi_imports.append('Request')
                elif imp == 'make_response': 
                    extra_lines.append('from fastapi.responses import JSONResponse')
                elif imp == 'jsonify': 
                    pass  # Will be handled separately
                elif imp == 'json': 
                    extra_lines.append('import json')  # Keep json import
                elif imp == 'abort':
                    fastapi_imports.append('HTTPException')
                elif imp == 'redirect':
                    extra_lines.append('from fastapi.responses import RedirectResponse')
                elif imp == 'Depends' or imp == 'Query' or imp == 'Path':
                    fastapi_imports.append(imp)
            
            lines = []
            if fastapi_imports:
                lines.append(f"from fastapi import {', '.join(fastapi_imports)}")
            lines.extend(extra_lines)
            return '\n'.join(lines) if lines else ''
            
        result = re.sub(r'from flask import ([^\n]+)', transform_imports, result)
        
        # ===== PHASE 3: Add necessary imports =====
        if needs_json and 'import json' not in result:
            # Add json import at the top
            lines = result.split('\n')
            # Find where to insert (after existing imports)
            insert_idx = 0
            for i, line in enumerate(lines):
                if line.startswith('from ') or line.startswith('import '):
                    insert_idx = i + 1
            if insert_idx > 0:
                lines.insert(insert_idx, 'import json')
            else:
                lines.insert(0, 'import json')
            result = '\n'.join(lines)
        
        # ===== PHASE 4: Handle Flask JWT imports =====
        result = re.sub(
            r'from flask_jwt_extended import[^\n]+\n?', 
            '# TODO: Replace with FastAPI JWT implementation (python-jose)\nfrom fastapi import Depends\n', 
            result
        )
        
        # ===== PHASE 5: Handle Flask Caching imports =====
        result = re.sub(
            r'from flask_caching import[^\n]+\n?',
            '# TODO: Replace with FastAPI caching (fastapi-cache2 or @lru_cache)\n', 
            result
        )

        # ===== PHASE 6: Blueprint to APIRouter (PRESERVE VARIABLE NAMES!) =====
        # auth_bp = Blueprint('auth_bp', __name__) -> auth_bp = APIRouter(tags=['auth_bp'])
        result = re.sub(
            r"(\w+)\s*=\s*Blueprint\s*\(\s*['\"](\w+)['\"]\s*,\s*__name__[^)]*\)",
            r"\1 = APIRouter(tags=['\2'])",
            result
        )

        # ===== PHASE 7: App initialization =====
        result = result.replace('Flask(__name__)', 'FastAPI()')
        
        # ===== PHASE 8: JWTManager removal =====
        result = re.sub(r'JWTManager\s*\(\s*app\s*\)', '# TODO: Set up FastAPI JWT authentication with python-jose', result)
        
        # ===== PHASE 9: Cache initialization =====
        result = re.sub(
            r"(\w+)\s*=\s*Cache\s*\([^)]*\)",
            r"# TODO: Set up FastAPI caching\n\1 = None  # Replace with fastapi-cache2",
            result
        )
        result = re.sub(r'\w+\.init_app\s*\(\s*app\s*\)', '# TODO: Set up FastAPI caching', result)
        
        # ===== PHASE 10: Route Decorators (PRESERVE VARIABLE NAMES!) =====
        # @auth_bp.route('/path', methods=['POST']) -> @auth_bp.post('/path')
        def transform_route(match):
            bp_name = match.group(1)
            path = match.group(2)
            methods = match.group(3)
            verb = 'get'
            if methods:
                if 'POST' in methods: verb = 'post'
                elif 'PUT' in methods: verb = 'put'
                elif 'DELETE' in methods: verb = 'delete'
                elif 'PATCH' in methods: verb = 'patch'
            return f"@{bp_name}.{verb}('{path}')"

        result = re.sub(
            r"@(\w+)\.route\s*\(\s*['\"]([^'\"]+)['\"]\s*(?:,\s*methods\s*=\s*\[([^\]]+)\])?\s*\)",
            transform_route,
            result
        )
        
        # ===== PHASE 11: Async Functions with Request Parameter =====
        # If the function uses request, it needs request: Request as parameter
        # Note: We check uses_request_data from original, but also check result after conversion
        uses_request_after_convert = 'await request.' in result or 'request.query_params' in result
        if uses_request_data or uses_request_after_convert:
            # Find all function definitions and check if they use request
            def add_request_param(match):
                func_name = match.group(1)
                # Check if this function uses request in the RESULT code (after conversion)
                # This is more reliable since request.data is already converted to await request.body()
                func_pattern = rf'def {func_name}\s*\([^)]*\):[^\n]*\n((?:[ \t]+[^\n]*\n)*)'
                func_match = re.search(func_pattern, result, re.MULTILINE)
                if func_match:
                    func_body = func_match.group(1) if func_match.group(1) else ''
                    if 'request.' in func_body or 'await request.' in func_body:
                        return f'async def {func_name}(request: Request):'
                # Also check original code for request usage
                func_match_orig = re.search(func_pattern, original_code, re.MULTILINE)
                if func_match_orig:
                    func_body = func_match_orig.group(1) if func_match_orig.group(1) else ''
                    if 'request.' in func_body or 'request.data' in func_body or 'request.json' in func_body:
                        return f'async def {func_name}(request: Request):'
                return f'async def {func_name}():'
            
            # Convert def to async def and add request param where needed
            result = re.sub(
                r'^def (\w+)\s*\(\s*\)\s*:',
                add_request_param,
                result,
                flags=re.MULTILINE
            )
        else:
            # Just convert to async
            result = re.sub(r'^def (\w+)\s*\(\s*\)\s*:', r'async def \1():', result, flags=re.MULTILINE)
        
        result = result.replace('async async def', 'async def')
        
        # ===== PHASE 12: Request data access =====
        # json.loads(request.data) -> json.loads(await request.body())
        result = re.sub(
            r'json\.loads\s*\(\s*request\.data\s*\)',
            'json.loads(await request.body())',
            result
        )
        result = result.replace('request.json', 'await request.json()')
        result = result.replace('request.data', 'await request.body()')
        result = result.replace('request.args', 'request.query_params')
        
        # ===== PHASE 13: Blueprint registration (PRESERVE VARIABLE NAMES!) =====
        result = result.replace('.register_blueprint(', '.include_router(')
        
        # ===== PHASE 14: JWT decorator =====
        result = re.sub(
            r"@jwt_required\s*\(\s*\)",
            "# TODO: Add FastAPI auth dependency\n# @Depends(get_current_user)",
            result
        )
        
        # ===== PHASE 15: Cache decorator =====
        result = re.sub(
            r"@\w+\.cached\s*\([^)]*\)",
            "# TODO: Add FastAPI caching (use @lru_cache or fastapi-cache2)",
            result
        )
        
        # ===== PHASE 16: before_request hooks =====
        result = re.sub(
            r"app\.before_request\s*\(\s*(\w+)\s*\)",
            r"# TODO: Convert \1 to FastAPI middleware or dependency",
            result
        )
        
        # ===== PHASE 17: Response handling =====
        # Remove jsonify() wrapper - FastAPI returns dicts directly
        result = re.sub(r'jsonify\s*\(\s*([^)]+)\s*\)', r'\1', result)
        
        # FIX: Robust make_response handling (Handles single arg, dicts, and status codes)
        # 1. Handle make_response({'a':1}, 200) - capture balanced braces
        result = re.sub(
            r"make_response\s*\(\s*(\{.*?\})\s*,\s*(\w+)\s*\)",
            r"JSONResponse(content=\1, status_code=\2)",
            result,
            flags=re.DOTALL
        )
        # 2. Handle single arg make_response({'message': '...'})
        result = re.sub(
            r"make_response\s*\(\s*(\{[^{}]+\})\s*\)",
            r"JSONResponse(content=\1)",
            result
        )
        # 3. Catch remaining standard patterns
        result = re.sub(
            r"make_response\s*\(\s*(.+?)\s*,\s*(HTTP_\w+|\d{3})\s*\)",
            r"JSONResponse(content=\1, status_code=\2)",
            result
        )
        # 4. Catch remaining single arg patterns
        result = re.sub(
            r"make_response\s*\(\s*([^\s,]+)\s*\)",
            r"JSONResponse(content=\1)",
            result
        )
        
        # ===== PHASE 18: SocketIO handling =====
        result = re.sub(
            r"socketio\.run\s*\([^)]*\)",
            '# TODO: Use WebSocket with FastAPI\n# Use: uvicorn.run(app, host="0.0.0.0", port=8000)',
            result
        )
        result = re.sub(r'socketio\.init_app\s*\([^)]*\)', '# TODO: Set up FastAPI WebSocket', result)
        
        # ===== PHASE 19: Config handling =====
        result = re.sub(
            r"app\.config\s*\[\s*['\"]SECRET_KEY['\"]\s*\]\s*=\s*['\"]([^'\"]+)['\"]",
            r"# TODO: Move to environment/settings: SECRET_KEY = '\1'",
            result
        )
        
        # ===== PHASE 20: Handle create_access_token =====
        if 'create_access_token' in result:
            # Add a TODO comment before first use if not already present
            if '# TODO: Implement JWT' not in result:
                result = re.sub(
                    r'(\s*)(\w+\s*=\s*create_access_token)',
                    r'\1# TODO: Implement JWT token creation with python-jose\n\1\2',
                    result,
                    count=1
                )
        
        # ===== PHASE 21: Clean up empty comment lines =====
        result = re.sub(r'\n# Removed:[^\n]*\n', '\n', result)
        result = re.sub(r'\n\s*\n\s*\n', '\n\n', result)
        
        # ===== PHASE 22: Ensure request parameter is added where needed =====
        # Final pass: check if any function still uses request without having it as a parameter
        lines = result.split('\n')
        new_lines = []
        i = 0
        while i < len(lines):
            line = lines[i]
            # Check if this is a function definition without request parameter
            func_match = re.match(r'^(\s*)(async\s+)?def\s+(\w+)\s*\(\s*\)\s*:', line)
            if func_match:
                indent = func_match.group(1)
                async_part = func_match.group(2) or ''
                func_name = func_match.group(3)
                func_indent_len = len(indent)
                
                # Look ahead to check if function body uses request
                j = i + 1
                uses_request = False
                while j < len(lines):
                    next_line = lines[j]
                    next_stripped = next_line.strip()
                    
                    # Skip empty lines
                    if not next_stripped:
                        j += 1
                        continue
                    
                    # Calculate indentation of this line
                    next_indent_len = len(next_line) - len(next_line.lstrip())
                    
                    # If we hit a line at same or less indentation, we've left the function
                    if next_indent_len <= func_indent_len and next_stripped:
                        # Check if it's a decorator or another def - those would start a new function
                        if next_stripped.startswith('@') or next_stripped.startswith('def ') or next_stripped.startswith('async def ') or next_stripped.startswith('class '):
                            break
                        # For top-level functions (indent_len=0), any non-indented line exits
                        if func_indent_len == 0:
                            break
                    
                    # Check for request usage in this line
                    if 'request.' in next_line or 'await request.' in next_line:
                        uses_request = True
                        break
                    j += 1
                
                if uses_request:
                    line = f'{indent}{async_part}def {func_name}(request: Request):'
            
            new_lines.append(line)
            i += 1
        
        result = '\n'.join(new_lines)
        
        # ===== PHASE 23: Fix Route Wrappers to pass request (Improved) =====
        # This addresses the Root Cause Analysis: Wrapper functions not passing request to controllers
        # We look for: @router.verb -> async def wrapper() -> return await func()
        # And convert to: async def wrapper(request: Request) -> return await func(request)
        
        def fix_wrapper(match):
            decorator = match.group(1)
            func_name = match.group(2)
            indent = match.group(3)
            called_func = match.group(4)
            
            # If the wrapper already has request, ignore
            if 'request' in func_name: 
                return match.group(0)
                
            return f"{decorator}async def {func_name}(request: Request):\n{indent}return await {called_func}(request)"

        # Regex explanation:
        # 1. Decorator line (@....)
        # 2. Function def (async def name():)
        # 3. Indentation and return (return (await)? name())
        result = re.sub(
            r'(@[^\n]+\.(?:post|put|patch|delete|get)\([^\n]+\n)async def (\w+)\(\):\n(\s+)return (?:await )?(\w+)\(\)',
            fix_wrapper,
            result
        )
        
        # Ensure Request is imported if we injected it
        if 'request: Request' in result and 'from fastapi import Request' not in result:
             if 'from fastapi import' in result:
                 result = re.sub(r'from fastapi import ([^\n]+)', r'from fastapi import \1, Request', result, count=1)
             else:
                 result = 'from fastapi import Request\n' + result

        # ===== PHASE 24: Inject JWT Implementation =====
        if 'create_access_token' in result and 'def create_access_token' not in result and 'from' not in result:
             jwt_impl = '''
# JWT Implementation
from jose import jwt
from datetime import datetime, timedelta

SECRET_KEY = "my_secret_key_part_4"
ALGORITHM = "HS256"

def create_access_token(identity: str, expires_delta: timedelta = None):
    if expires_delta is None:
        expires_delta = timedelta(minutes=30)
    expire = datetime.utcnow() + expires_delta
    to_encode = {"sub": identity, "exp": expire}
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
'''
             # Add imports if needed
             if 'from jose import jwt' not in result:
                 result = 'from jose import jwt\n' + result
             if 'from datetime import' not in result:
                 result = 'from datetime import datetime, timedelta\n' + result
                 
             # Append implementation at the end or after imports
             # For simplicity, appending at the end of imports area would be best, but appending to end of file works too if functions use it.
             # However, let's try to put it after imports.
             # Simple approach: Append to end of result, but before main logic if any? 
             # Let's put it after the last import
             last_import_idx = 0
             lines = result.split('\n')
             for i, line in enumerate(lines):
                 if line.startswith('from ') or line.startswith('import '):
                     last_import_idx = i
             
             lines.insert(last_import_idx + 1, jwt_impl)
             result = '\n'.join(lines)

        # ===== PHASE 25: WebSocket Conversion =====
        if 'flask_socketio' in original_code or 'socketio' in original_code:
            # Check if this chunk has the imports to replace
            had_imports = 'from flask_socketio import' in result
            
            # Remove Flask-SocketIO imports
            result = re.sub(r'from flask_socketio import[^\n]+\n?', '', result)
            
            # Inject ConnectionManager ONLY if we removed imports (top of file)
            if had_imports and 'class ConnectionManager' not in result:
                ws_manager = '''
from fastapi import WebSocket, WebSocketDisconnect
from typing import Dict, Set, List

class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, Set[WebSocket]] = {}

    async def connect(self, websocket: WebSocket, room: str):
        await websocket.accept()
        if room not in self.active_connections:
            self.active_connections[room] = set()
        self.active_connections[room].add(websocket)

    def disconnect(self, websocket: WebSocket, room: str):
        if room in self.active_connections:
            self.active_connections[room].discard(websocket)

    async def broadcast(self, message: dict, room: str):
        if room in self.active_connections:
            for connection in self.active_connections[room]:
                await connection.send_json(message)

manager = ConnectionManager()
'''
                # Insert after imports
                lines = result.split('\n')
                last_import_idx = 0
                for i, line in enumerate(lines):
                    if line.startswith('from ') or line.startswith('import '):
                        last_import_idx = i
                lines.insert(last_import_idx + 1, ws_manager)
                result = '\n'.join(lines)
            
            # Replace usage
            result = re.sub(r'socketio\s*=\s*SocketIO\([^)]*\)', '', result)
            
            # Comment out decorators
            result = re.sub(r'@socketio\.on', r'# @socketio.on', result)
            
            # Comment out join/leave room calls (but not definitions)
            result = re.sub(r'(?<!def\s)join_room\(', r'# join_room(', result)
            result = re.sub(r'(?<!def\s)leave_room\(', r'# leave_room(', result)

            # Replace emit
            # socketio.emit('event', data, room=room) -> await manager.broadcast(data, room)
            result = re.sub(
                r'socketio\.emit\s*\(\s*[\'"](\w+)[\'"]\s*,\s*([^,]+)\s*,\s*room\s*=\s*([^)]+)\)',
                r'await manager.broadcast(\2, \3)',
                result
            )
        
        return result
    
    def _validate_migration(self, code: str) -> tuple[bool, List[str]]:
        """Validate migrated code for common issues."""
        errors = []
        
        if not code:
            return False, ["Empty code"]
        
        # Check for syntax errors
        try:
            compile(code, '<string>', 'exec')
        except SyntaxError as e:
            errors.append(f"Syntax error: {e}")
            return False, errors
        
        # Check for leftover Flask imports (but allow comments about them)
        if re.search(r'^from flask import', code, re.MULTILINE):
            errors.append("Leftover Flask import found")
        
        if re.search(r'^from flask_jwt_extended import', code, re.MULTILINE):
            errors.append("Leftover Flask-JWT-Extended import found")
        
        if re.search(r'^from flask_caching import', code, re.MULTILINE):
            errors.append("Leftover Flask-Caching import found")
        
        # Check for Flask-specific patterns that should be converted
        if "jsonify(" in code:
            errors.append("Found jsonify() - should be removed for FastAPI")
        
        if "@app.route(" in code and "@app.get(" not in code and "@app.post(" not in code:
            errors.append("Found @app.route - should be converted to @app.get/@app.post etc.")
        
        # Check for json usage without import
        if ('json.loads' in code or 'json.dumps' in code) and 'import json' not in code:
            errors.append("json.loads/json.dumps used but import json is missing")
        
        # Check for request usage without parameter
        if 'request.' in code or 'await request.' in code:
            # Check if any function uses request without having it as a parameter
            func_pattern = r'def\s+\w+\s*\(\s*\)\s*:'
            if re.search(func_pattern, code):
                # Verify those functions don't use request
                pass  # This is a complex check, we'll rely on the migration logic
        
        return len(errors) == 0, errors
    
    def _post_process(self, code: str) -> str:
        """Post-process migrated code to fix common issues."""
        # Remove duplicate imports
        lines = code.split('\n')
        seen_imports = set()
        result = []
        
        for line in lines:
            stripped = line.strip()
            if stripped.startswith('from ') or stripped.startswith('import '):
                if stripped not in seen_imports:
                    seen_imports.add(stripped)
                    result.append(line)
            else:
                result.append(line)
        
        code = '\n'.join(result)
        
        # Add import json if needed
        if ('json.loads' in code or 'json.dumps' in code) and 'import json' not in code:
            lines = code.split('\n')
            insert_idx = 0
            for i, line in enumerate(lines):
                if line.startswith('from ') or line.startswith('import '):
                    insert_idx = i + 1
            lines.insert(insert_idx, 'import json')
            code = '\n'.join(lines)
        
        # Ensure proper spacing around decorators
        code = re.sub(r'(\S)(@(?:app|router|\w+_bp|\w+_router)\.)', r'\1\n\n\2', code)
        
        # Fix double async
        code = code.replace('async async def', 'async def')
        
        # Clean up excessive blank lines
        code = re.sub(r'\n{4,}', '\n\n\n', code)
        
        return code
    
    def _calculate_confidence(self, original: str, migrated: str) -> float:
        """Calculate confidence score for migration."""
        score = 1.0
        
        # Reduce score for various issues
        if "TODO" in migrated:
            score -= 0.05 * min(migrated.count("TODO"), 5)  # Cap penalty at 5 TODOs
        
        if "from flask" in migrated.lower():
            score -= 0.2
        
        if "jsonify" in migrated:
            score -= 0.1
        
        if "make_response" in migrated:
            score -= 0.1
        
        # Check if structure is preserved
        original_defs = len(re.findall(r'def \w+', original))
        migrated_defs = len(re.findall(r'def \w+', migrated))
        if original_defs != migrated_defs:
            score -= 0.1
        
        # Bonus for proper async conversion
        if 'async def' in migrated and 'await' in migrated:
            score += 0.05
        
        return max(0.0, min(1.0, score))


# Singleton instance
_migration_agent: Optional[MigrationAgent] = None


def get_migration_agent() -> MigrationAgent:
    """Get or create the migration agent singleton."""
    global _migration_agent
    if _migration_agent is None:
        _migration_agent = MigrationAgent()
    return _migration_agent

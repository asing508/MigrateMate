"""Code parser for extracting functions, classes, and dependencies."""

import ast
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field

class CodeParser:
    def parse_code(self, source_code: str, file_path: str) -> List[Dict[str, Any]]:
        """
        Parse Python code to extract functions, classes, and GLOBAL content (imports, variables).
        Returns a list of chunks that covers the ENTIRE file content in order.
        """
        chunks = []
        try:
            tree = ast.parse(source_code)
            lines = source_code.splitlines(keepends=True)
            total_lines = len(lines)
            
            # 1. Identify specific nodes we want to migrate specifically (Functions, Classes)
            nodes = []
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    # We want top-level or method nodes. 
                    # Note: ast.walk explores children. We might want to avoid nested functions 
                    # if we treat the parent function as one block.
                    # For now, let's just collect all top-level-ish nodes.
                    # Or better: let's rely on the fact that if we migrate a class, we migrate its methods inside it.
                    # So we should filter out nodes that are children of other collected nodes.
                    nodes.append(node)

            # Sort nodes by start line
            nodes.sort(key=lambda x: x.lineno)

            # Filter nested nodes
            root_nodes = []
            if nodes:
                curr = nodes[0]
                root_nodes.append(curr)
                for next_node in nodes[1:]:
                    # If next_node starts after current ends, it's a new root
                    # (AST end_lineno covers the whole block)
                    if hasattr(curr, 'end_lineno') and next_node.lineno > curr.end_lineno:
                        root_nodes.append(next_node)
                        curr = next_node
                    else:
                        # It's nested or overlapping, skip it (it's part of the current root)
                        pass
            
            # 2. Iterate and create chunks, filling in gaps
            current_line = 0 # 0-indexed for list slicing
            
            for node in root_nodes:
                # Capture gap before node
                node_start_line = node.lineno - 1 
                # Check decorators
                if node.decorator_list:
                    node_start_line = node.decorator_list[0].lineno - 1
                
                if node_start_line > current_line:
                    gap_content = "".join(lines[current_line:node_start_line])
                    if gap_content.strip(): # Only add non-empty gaps or important whitespace
                        chunks.append({
                            "name": "global_code",
                            "type": "misc",
                            "content": gap_content,
                            "start_line": current_line + 1,
                            "end_line": node_start_line,
                            "file_path": file_path,
                            "chunk_type": "misc"
                        })
                    elif gap_content: # Preserve whitespace
                        chunks.append({
                            "name": "whitespace",
                            "type": "misc",
                            "content": gap_content,
                            "start_line": current_line + 1,
                            "end_line": node_start_line,
                            "file_path": file_path,
                            "chunk_type": "ignored" 
                        })

                # Capture node content
                node_end_line = getattr(node, 'end_lineno', node_start_line + 1)
                node_content = "".join(lines[node_start_line:node_end_line])
                
                chunks.append({
                    "name": node.name,
                    "type": "class" if isinstance(node, ast.ClassDef) else "function",
                    "content": node_content,
                    "start_line": node_start_line + 1,
                    "end_line": node_end_line,
                    "file_path": file_path,
                    "chunk_type": "class" if isinstance(node, ast.ClassDef) else "function"
                })
                
                current_line = node_end_line

            # Capture tail
            if current_line < total_lines:
                tail_content = "".join(lines[current_line:])
                chunks.append({
                    "name": "global_code_tail",
                    "type": "misc",
                    "content": tail_content,
                    "start_line": current_line + 1,
                    "end_line": total_lines,
                    "file_path": file_path,
                    "chunk_type": "misc"
                })

        except Exception as e:
            print(f"Error parsing {file_path}: {e}")
            # Fallback: return whole file as one chunk
            chunks = [{
                "name": "whole_file",
                "type": "misc",
                "content": source_code,
                "start_line": 1,
                "end_line": len(source_code.splitlines()),
                "file_path": file_path,
                "chunk_type": "misc"
            }]
            
        return chunks


@dataclass
class CodeChunk:
    name: str
    chunk_type: str
    start_line: int
    end_line: int
    content: str
    file_path: str
    dependencies: List[str] = field(default_factory=list)

def parse_python_file(content: str, file_path: str) -> List[CodeChunk]:
    """Parse python code and return CodeChunk objects."""
    raw_chunks = code_parser.parse_code(content, file_path)
    return [
        CodeChunk(
            name=c["name"],
            chunk_type=c["chunk_type"],
            start_line=c["start_line"],
            end_line=c["end_line"],
            content=c["content"],
            file_path=c["file_path"],
            dependencies=[] # TODO: Implement dependency extraction
        )
        for c in raw_chunks
    ]

def detect_flask_routes(code: str) -> List[Dict[str, Any]]:
    """Detect Flask routes in code."""
    routes = []
    try:
        tree = ast.parse(code)
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                for decorator in node.decorator_list:
                    if isinstance(decorator, ast.Call):
                        func = decorator.func
                        # Match @app.route or @bp.route
                        if isinstance(func, ast.Attribute) and func.attr == 'route':
                            path = "unknown"
                            if decorator.args:
                                arg = decorator.args[0]
                                if isinstance(arg, ast.Constant):
                                    path = arg.value
                                elif isinstance(arg, ast.Str):
                                    path = arg.s
                            
                            methods = ["GET"]
                            for kw in decorator.keywords:
                                if kw.arg == "methods" and isinstance(kw.value, ast.List):
                                    methods = [
                                        elt.value if isinstance(elt, ast.Constant) else elt.s 
                                        for elt in kw.value.elts 
                                        if isinstance(elt, (ast.Constant, ast.Str))
                                    ]
                            
                            routes.append({
                                "path": path,
                                "methods": methods,
                                "function_name": node.name,
                                "line_number": node.lineno,
                                "framework": "flask"
                            })
    except Exception as e:
        print(f"Error extracting flask routes: {e}")
    return routes

def detect_fastapi_routes(code: str) -> List[Dict[str, Any]]:
    """Detect FastAPI routes in code."""
    routes = []
    methods_set = {"get", "post", "put", "delete", "patch", "options", "head", "trace"}
    try:
        tree = ast.parse(code)
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                for decorator in node.decorator_list:
                    if isinstance(decorator, ast.Call):
                        func = decorator.func
                        # Match @app.get, @router.post, etc.
                        if isinstance(func, ast.Attribute) and func.attr in methods_set:
                            path = "unknown"
                            if decorator.args:
                                arg = decorator.args[0]
                                if isinstance(arg, ast.Constant):
                                    path = arg.value
                                elif isinstance(arg, ast.Str):
                                    path = arg.s
                            
                            routes.append({
                                "path": path,
                                "methods": [func.attr.upper()],
                                "function_name": node.name,
                                "line_number": node.lineno,
                                "framework": "fastapi"
                            })
    except Exception as e:
        print(f"Error extracting fastapi routes: {e}")
    return routes

code_parser = CodeParser()
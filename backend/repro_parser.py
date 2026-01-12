
import ast

code = """
@app.route('/')
def home():
    return "Hello"

@app.route('/users', methods=['GET'])
def get_users():
    return "Users"
"""

tree = ast.parse(code)
lines = code.splitlines(keepends=True)

for node in ast.walk(tree):
    if isinstance(node, ast.FunctionDef):
        print(f"--- Node: {node.name} ---")
        print(f"node.lineno: {node.lineno}")
        
        # 1. AST segment
        content_ast = ast.get_source_segment(code, node)
        print(f"AST Segment:\n{content_ast!r}")
        
        # 2. Manual slicing (fallback logic)
        start = node.lineno - 1
        end = node.end_lineno
        content_slice = "".join(lines[start:end])
        print(f"Manual Slice:\n{content_slice!r}")

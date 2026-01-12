
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
        
        start_line = node.lineno
        if node.decorator_list:
            start_line = node.decorator_list[0].lineno
            print(f"Has decorators. Start line adjusted from {node.lineno} to {start_line}")
        
        start = start_line - 1
        end = node.end_lineno
        content = "".join(lines[start:end])
        print(f"Corrected Content:\n{content!r}")

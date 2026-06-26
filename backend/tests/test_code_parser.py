"""Tests for the AST-based code parser."""

from app.services.code_parser import (
    parse_python_file, detect_flask_routes, detect_fastapi_routes,
)


SAMPLE = '''import os

def hello():
    return "hi"

class Greeter:
    def greet(self, name):
        return f"Hello {name}"

x = 1
'''


def test_parse_covers_whole_file_in_order():
    chunks = parse_python_file(SAMPLE, "sample.py")
    names = [c.name for c in chunks]
    # Functions/classes are recognised; surrounding code is preserved as chunks.
    assert "hello" in names
    assert "Greeter" in names
    # Reassembling the chunk contents reproduces the original source.
    assert "".join(c.content for c in chunks) == SAMPLE


def test_nested_methods_not_emitted_as_top_level_chunks():
    chunks = parse_python_file(SAMPLE, "sample.py")
    func_class = [c for c in chunks if c.chunk_type in ("function", "class")]
    names = [c.name for c in func_class]
    # `greet` is a method of Greeter, not a separate top-level chunk.
    assert "greet" not in names


def test_parse_invalid_syntax_falls_back_to_whole_file():
    chunks = parse_python_file("def broken(:\n  pass", "bad.py")
    assert len(chunks) == 1
    assert chunks[0].name == "whole_file"


def test_detect_flask_routes():
    code = '''
@app.route("/users", methods=["GET", "POST"])
def users():
    return "ok"
'''
    routes = detect_flask_routes(code)
    assert len(routes) == 1
    assert routes[0]["path"] == "/users"
    assert set(routes[0]["methods"]) == {"GET", "POST"}


def test_detect_fastapi_routes():
    code = '''
@router.post("/login")
async def login():
    return {}
'''
    routes = detect_fastapi_routes(code)
    assert len(routes) == 1
    assert routes[0]["path"] == "/login"
    assert routes[0]["methods"] == ["POST"]

"""
Test script for local directory migration.
Run this from the backend directory after starting the server:
    uvicorn main:app --reload --port 8000
"""

import requests
import json
import shutil
import os

API_URL = "http://localhost:8000/api/v1"
SOURCE_DIR = r"D:\Social-Media-Backend-main\src"
OUTPUT_DIR = r"D:\Social-Media-Backend_fastapi_new"

def clear_output_dir():
    """Clear the output directory if it exists."""
    if os.path.exists(OUTPUT_DIR):
        print(f"🗑️  Clearing existing output directory: {OUTPUT_DIR}")
        shutil.rmtree(OUTPUT_DIR)

def run_migration():
    """Run the local directory migration."""
    print(f"🚀 MigrateMate - Local Directory Migration")
    print(f"=" * 60)
    print(f"📂 Source: {SOURCE_DIR}")
    print(f"📂 Output: {OUTPUT_DIR}")
    print(f"=" * 60)
    
    # Clear previous output
    clear_output_dir()
    
    # Run migration
    print("\n⏳ Starting migration...")
    response = requests.post(
        f"{API_URL}/batch/local",
        json={
            "source_dir": SOURCE_DIR,
            "output_dir": OUTPUT_DIR,
            "source_framework": "flask",
            "target_framework": "fastapi"
        },
        timeout=300  # 5 minute timeout for large projects
    )
    
    if response.status_code != 200:
        print(f"\n❌ Migration failed!")
        print(f"   Status: {response.status_code}")
        print(f"   Response: {response.text}")
        return False
    
    result = response.json()
    
    # Print results
    print(f"\n✅ Migration completed!")
    print(f"\n{'=' * 60}")
    print(f"📊 Summary:")
    print(f"{'=' * 60}")
    summary = result.get('summary', {})
    print(f"   Total files:      {summary.get('total_files', 0)}")
    print(f"   Files migrated:   {summary.get('files_migrated', 0)}")
    print(f"   Total chunks:     {summary.get('total_chunks', 0)}")
    print(f"   Chunks succeeded: {summary.get('chunks_succeeded', 0)}")
    print(f"   Chunks failed:    {summary.get('chunks_failed', 0)}")
    print(f"   Avg confidence:   {summary.get('average_confidence', 0):.2%}")
    
    print(f"\n📁 Migrated files:")
    for file in result.get('files', []):
        confidence = file.get('confidence', 0)
        status = "✅" if confidence > 0.7 else "⚠️" if confidence > 0.5 else "❌"
        print(f"   {status} {file.get('path')} (confidence: {confidence:.2%})")
    
    print(f"\n📂 Output directory: {result.get('output_dir')}")
    print(f"\n🚀 To run the migrated project:")
    print(f"   cd {OUTPUT_DIR}")
    print(f"   pip install -r requirements.txt")
    print(f"   uvicorn app:app --reload")
    
    return True

def check_server():
    """Check if the server is running."""
    try:
        response = requests.get(f"{API_URL.replace('/api/v1', '')}/health")
        if response.status_code == 200:
            return True
    except requests.exceptions.ConnectionError:
        pass
    return False

if __name__ == "__main__":
    if not check_server():
        print("❌ Server is not running!")
        print("   Please start the server first:")
        print("   uvicorn main:app --reload --port 8000")
        exit(1)
    
    run_migration()

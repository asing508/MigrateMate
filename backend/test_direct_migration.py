"""
Direct test script that tests migration without needing the API server.
Run from backend directory: python test_direct_migration.py
"""

import asyncio
import os
import sys

# Add the app directory to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.services.migration_service import get_batch_migration_service

SOURCE_DIR = r"D:\Social-Media-Backend-main\src"
OUTPUT_DIR = r"D:\Social-Media-Backend_fastapi_test"


async def run_migration():
    """Run local directory migration directly."""
    print("🚀 MigrateMate - Direct Migration Test")
    print("=" * 60)
    print(f"📂 Source: {SOURCE_DIR}")
    print(f"📂 Output: {OUTPUT_DIR}")
    print("=" * 60)
    
    # Clean output directory
    import shutil
    if os.path.exists(OUTPUT_DIR):
        print(f"\n🗑️  Clearing existing output directory...")
        shutil.rmtree(OUTPUT_DIR)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Get service and run migration
    service = get_batch_migration_service()
    
    print("\n⏳ Starting migration...")
    result = await service.migrate_local_directory(
        source_dir=SOURCE_DIR,
        output_dir=OUTPUT_DIR,
        source_framework="flask",
        target_framework="fastapi"
    )
    
    # Print results
    print("\n✅ Migration completed!")
    print("\n" + "=" * 60)
    print("📊 Summary:")
    print("=" * 60)
    summary = result.get('summary', {})
    print(f"   Total files:      {summary.get('total_files', 0)}")
    print(f"   Files migrated:   {summary.get('files_migrated', 0)}")
    print(f"   Total chunks:     {summary.get('total_chunks', 0)}")
    print(f"   Chunks succeeded: {summary.get('chunks_succeeded', 0)}")
    print(f"   Chunks failed:    {summary.get('chunks_failed', 0)}")
    print(f"   Avg confidence:   {summary.get('average_confidence', 0):.2%}")
    
    print(f"\n📁 Output directory: {result.get('output_dir')}")
    
    return result


def check_output():
    """Check key output files for issues."""
    print("\n" + "=" * 60)
    print("🔍 Checking output files...")
    print("=" * 60)
    
    issues = []
    
    # Check app.py 
    app_py = os.path.join(OUTPUT_DIR, "app.py")
    if os.path.exists(app_py):
        with open(app_py, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check for router collision
        router_imports = content.count("from routes.") 
        router_count = content.count("import router")
        if router_count > 1:
            issues.append(f"❌ app.py: Multiple 'import router' statements ({router_count}) - COLLISION!")
        else:
            print(f"✅ app.py: No router import collision")
        
        # Check for proper blueprint names
        if "import auth_bp" in content:
            print(f"✅ app.py: auth_bp name preserved")
        else:
            issues.append(f"⚠️  app.py: auth_bp name not found in imports")
    
    # Check user_controller.py
    controller = os.path.join(OUTPUT_DIR, "controllers", "user_controller.py")
    if os.path.exists(controller):
        with open(controller, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check for import json
        if 'import json' in content:
            print(f"✅ user_controller.py: import json present")
        elif 'json.loads' in content:
            issues.append(f"❌ user_controller.py: json.loads used without import json")
        
        # Check for request: Request parameter
        if 'def register(request: Request)' in content or 'async def register(request: Request)' in content:
            print(f"✅ user_controller.py: request: Request parameter present")
        elif 'request.body()' in content or 'await request.body()' in content:
            issues.append(f"❌ user_controller.py: request.body() used without request parameter")
        
        # Check syntax
        try:
            compile(content, controller, 'exec')
            print(f"✅ user_controller.py: No syntax errors")
        except SyntaxError as e:
            issues.append(f"❌ user_controller.py: Syntax error: {e}")
    
    # Summary
    print("\n" + "=" * 60)
    if issues:
        print("❌ Issues found:")
        for issue in issues:
            print(f"   {issue}")
    else:
        print("✅ All checks passed!")
    print("=" * 60)
    
    return len(issues) == 0


if __name__ == "__main__":
    result = asyncio.run(run_migration())
    check_output()

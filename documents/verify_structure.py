import os
import sys

def verify():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    print(f"Verifying DRISHTI structure at: {base_dir}")
    
    expected_folders = [
        "drishti",
        "drishti/agents",
        "drishti/tools",
        "drishti/utils"
    ]
    
    expected_files = [
        "app.py",
        "drishti/__init__.py",
        "drishti/state.py",
        "drishti/graph.py",
        "drishti/agents/__init__.py",
        "drishti/agents/l1_classifier.py",
        "drishti/agents/l2_classifier.py",
        "drishti/agents/planner.py",
        "drishti/agents/executor.py",
        "drishti/agents/synthesizer.py",
        "drishti/tools/__init__.py",
        "drishti/tools/sql_tool.py",
        "drishti/tools/nosql_tool.py",
        "drishti/tools/vector_tool.py",
        "drishti/tools/viz_tool.py",
        "drishti/tools/validator.py",
        "drishti/utils/__init__.py",
        "drishti/utils/nlp.py",
        "drishti/utils/db_connections.py",
        "drishti/utils/audit_logger.py"
    ]
    
    failures = 0
    
    # 1. Verify Folders
    for folder in expected_folders:
        folder_path = os.path.join(base_dir, folder.replace("/", os.sep))
        if not os.path.isdir(folder_path):
            print(f"[FAIL] Missing folder: {folder}")
            failures += 1
        else:
            print(f"[OK] Folder exists: {folder}")
            
    # 2. Verify Files
    for file in expected_files:
        file_path = os.path.join(base_dir, file.replace("/", os.sep))
        if not os.path.isfile(file_path):
            print(f"[FAIL] Missing file: {file}")
            failures += 1
        else:
            print(f"[OK] File exists: {file}")
            
    # 3. Verify Importability
    sys.path.insert(0, base_dir)
    try:
        import drishti
        print("[OK] Core package 'drishti' is importable.")
    except Exception as e:
        print(f"[FAIL] Failed to import 'drishti': {e}")
        failures += 1
        
    if failures == 0:
        print("\nAll checks passed successfully! Folder structure matches the blueprint perfectly.")
        sys.exit(0)
    else:
        print(f"\nVerification failed with {failures} error(s).")
        sys.exit(1)

if __name__ == "__main__":
    verify()

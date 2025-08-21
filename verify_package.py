#!/usr/bin/env python3
"""
Package Verification Script for CAP3D-Viz

This script verifies that the package structure is correct and ready for publication.
"""

import sys
import os
from pathlib import Path

def verify_package_structure():
    """Verify that all required files and directories exist"""
    print("🔍 Verifying package structure...")
    
    required_files = [
        "cap3d_viz/__init__.py",
        "cap3d_viz/data_models.py", 
        "cap3d_viz/parser.py",
        "cap3d_viz/visualizer.py",
        "cap3d_viz/utils.py",
        "setup.py",
        "pyproject.toml",
        "requirements.txt",
        "README.md",
        "CITATION.cff",
        "LICENSE",
        "CHANGELOG.md",
    ]
    
    missing_files = []
    for file_path in required_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
        else:
            print(f"✅ {file_path}")
    
    if missing_files:
        print(f"\n❌ Missing files: {missing_files}")
        return False
    
    print("✅ All required files present")
    return True

def verify_imports():
    """Verify that the package imports correctly"""
    print("\n🔍 Verifying package imports...")
    
    try:
        import cap3d_viz
        print(f"✅ Package version: {cap3d_viz.__version__}")
        
        # Test core imports
        from cap3d_viz import OptimizedCap3DVisualizer
        from cap3d_viz import StreamingCap3DParser
        from cap3d_viz import Block, Layer, PolyElement
        from cap3d_viz import load_and_visualize, quick_preview
        
        print("✅ All core imports successful")
        
        # Test object creation
        visualizer = OptimizedCap3DVisualizer()
        print("✅ Visualizer object creation successful")
        
        # Test Block creation
        block = Block(
            name="test", type="conductor", parent_name="medium",
            base=[0,0,0], v1=[1,0,0], v2=[0,1,0], hvec=[0,0,1]
        )
        print("✅ Block object creation successful")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def verify_documentation():
    """Verify documentation files exist and are readable"""
    print("\n🔍 Verifying documentation...")
    
    doc_files = [
        "docs/installation.md",
        "docs/performance.md", 
        "docs_publication/softwarex_paper_draft.md",
        "docs_publication/publication_roadmap.md"
    ]
    
    for doc_file in doc_files:
        if Path(doc_file).exists():
            print(f"✅ {doc_file}")
        else:
            print(f"❌ Missing: {doc_file}")
            return False
    
    return True

def verify_archive():
    """Verify legacy files are properly archived"""
    print("\n🔍 Verifying legacy file cleanup...")
    
    # Check that archive exists
    if not Path("archive_legacy").exists():
        print("❌ Archive directory missing")
        return False
    
    # Check that main directory is clean
    legacy_patterns = ["cap3d_matplotlib", "cap3d_plotly", "ehnanced_Cache_memory"]
    
    for pattern in legacy_patterns:
        if any(Path(".").glob(f"*{pattern}*")):
            print(f"❌ Legacy files still present: {pattern}")
            return False
    
    print("✅ Legacy files properly archived")
    return True

def main():
    """Main verification function"""
    print("🚀 CAP3D-Viz Package Verification")
    print("=" * 50)
    
    checks = [
        ("Package Structure", verify_package_structure),
        ("Python Imports", verify_imports), 
        ("Documentation", verify_documentation),
        ("Legacy Cleanup", verify_archive)
    ]
    
    all_passed = True
    
    for check_name, check_func in checks:
        try:
            if not check_func():
                all_passed = False
        except Exception as e:
            print(f"❌ {check_name} failed with error: {e}")
            all_passed = False
    
    print("\n" + "=" * 50)
    if all_passed:
        print("🎉 ALL VERIFICATION CHECKS PASSED!")
        print("✅ Package is ready for publication preparation")
        print("\n📋 Next steps:")
        print("1. Create public GitHub repository")
        print("2. Set up CI/CD pipeline") 
        print("3. Generate API documentation")
        print("4. Prepare benchmark datasets")
        print("5. Finalize SoftwareX paper")
    else:
        print("❌ Some verification checks failed")
        print("Please fix the issues before proceeding")
        sys.exit(1)

if __name__ == "__main__":
    main()

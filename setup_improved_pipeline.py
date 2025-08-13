#!/usr/bin/env python3
"""
Setup script for the improved BigCodeBench testing pipeline.
Installs required dependencies and checks system compatibility.
"""

import subprocess
import sys
import importlib
from pathlib import Path

def check_python_version():
    """Check if Python version is compatible."""
    if sys.version_info < (3, 7):
        print("❌ Python 3.7 or higher is required")
        return False
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
    return True

def install_package(package):
    """Install a single package using pip."""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        return True
    except subprocess.CalledProcessError:
        return False

def check_and_install_packages():
    """Check and install required packages."""
    required_packages = [
        "tqdm>=4.60.0",
        "psutil>=5.8.0",
        "numpy>=1.20.0",
        "pandas>=1.3.0",
        "matplotlib>=3.3.0",
        "scikit-learn>=1.0.0",
        "requests>=2.25.0",
        "beautifulsoup4>=4.9.0",
        "lxml>=4.6.0",
        "openpyxl>=3.0.0",
        "seaborn>=0.11.0",
        "flask>=2.0.0",
        "werkzeug>=2.0.0",
        "pillow>=8.0.0",
        "opencv-python>=4.5.0",
        "wordcloud>=1.8.0"
    ]
    
    print("📦 Installing required packages...")
    
    try:
        # Try to install all at once first
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", 
            "-r", "requirements-improved.txt"
        ])
        print("✅ All packages installed successfully!")
        return True
    except subprocess.CalledProcessError:
        print("⚠️  Batch install failed, trying individual packages...")
        
        failed_packages = []
        for package in required_packages:
            print(f"  Installing {package.split('>=')[0]}...")
            if not install_package(package):
                failed_packages.append(package)
                print(f"    ❌ Failed to install {package}")
            else:
                print(f"    ✅ Installed {package.split('>=')[0]}")
        
        if failed_packages:
            print(f"\n❌ Failed to install: {', '.join(failed_packages)}")
            print("Please install these manually or check your internet connection")
            return False
        
        print("✅ All packages installed successfully!")
        return True

def test_imports():
    """Test that all required packages can be imported."""
    test_packages = [
        ("tqdm", "tqdm"),
        ("psutil", "psutil"),
        ("numpy", "numpy"),
        ("pandas", "pandas"),
        ("matplotlib", "matplotlib.pyplot"),
        ("scikit-learn", "sklearn"),
        ("requests", "requests"),
        ("beautifulsoup4", "bs4"),
        ("lxml", "lxml"),
        ("openpyxl", "openpyxl"),
        ("seaborn", "seaborn"),
        ("flask", "flask"),
        ("werkzeug", "werkzeug"),
        ("pillow", "PIL"),
        ("opencv-python", "cv2"),
        ("wordcloud", "wordcloud")
    ]
    
    print("\n🧪 Testing package imports...")
    failed_imports = []
    
    for package_name, import_name in test_packages:
        try:
            importlib.import_module(import_name)
            print(f"  ✅ {package_name}")
        except ImportError as e:
            print(f"  ❌ {package_name}: {e}")
            failed_imports.append(package_name)
    
    if failed_imports:
        print(f"\n❌ Import failures: {', '.join(failed_imports)}")
        return False
    
    print("✅ All imports successful!")
    return True

def check_system_compatibility():
    """Check system compatibility and resources."""
    import multiprocessing
    import shutil
    
    print("\n🖥️  System compatibility check...")
    
    # Check CPU cores
    cpu_count = multiprocessing.cpu_count()
    print(f"  CPU cores: {cpu_count}")
    if cpu_count < 2:
        print("  ⚠️  Warning: Less than 2 CPU cores detected. Multithreading benefits will be limited.")
    else:
        print(f"  ✅ Good for parallel testing (recommended workers: {min(cpu_count, 8)})")
    
    # Check available disk space
    try:
        disk_usage = shutil.disk_usage(".")
        free_gb = disk_usage.free / (1024**3)
        print(f"  Free disk space: {free_gb:.1f} GB")
        if free_gb < 1:
            print("  ❌ Warning: Less than 1GB free disk space. Tests may fail.")
        else:
            print("  ✅ Sufficient disk space")
    except:
        print("  ⚠️  Could not check disk space")
    
    # Check memory (approximate)
    try:
        import psutil
        memory = psutil.virtual_memory()
        memory_gb = memory.total / (1024**3)
        print(f"  Total RAM: {memory_gb:.1f} GB")
        if memory_gb < 4:
            print("  ⚠️  Warning: Less than 4GB RAM. Consider using fewer worker threads.")
        else:
            print("  ✅ Sufficient memory")
    except:
        print("  ⚠️  Could not check memory")

def main():
    """Main setup function."""
    print("🚀 BigCodeBench Improved Pipeline Setup")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Install packages
    if not check_and_install_packages():
        print("\n❌ Package installation failed. Please check the errors above.")
        sys.exit(1)
    
    # Test imports
    if not test_imports():
        print("\n❌ Import tests failed. Please check the errors above.")
        sys.exit(1)
    
    # Check system compatibility
    check_system_compatibility()
    
    print("\n🎉 Setup completed successfully!")
    print("\nYou can now run the improved pipeline:")
    print("  python3 improved_test_pipeline.py --model o4-mini")
    print("\nOr with custom thread count:")
    print("  python3 improved_test_pipeline.py --model o4-mini --workers 12")
    print("\nFor detailed progress bars, tqdm is now installed and will be used automatically!")

if __name__ == "__main__":
    main()
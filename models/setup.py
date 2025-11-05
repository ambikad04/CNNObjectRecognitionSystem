import os
import subprocess
import sys

def create_directories():
    """Create necessary directories for the project."""
    directories = [
        'models',
        'weights',
        'data',
        'data/images',
        'data/masks',
        'data/annotations'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"Created directory: {directory}")

def install_requirements():
    """Install required packages."""
    requirements = [
        'torch>=2.0.0',
        'torchvision>=0.15.0',
        'numpy>=1.21.0',
        'opencv-python>=4.5.0',
        'faiss-cpu>=1.7.0',
        'scikit-learn>=0.24.0',
        'pillow>=8.0.0',
        'matplotlib>=3.4.0',
        'tqdm>=4.62.0'
    ]
    
    print("Installing requirements...")
    for package in requirements:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        print(f"Installed: {package}")

def main():
    print("Setting up the object detection project environment...")
    
    # Create directories
    create_directories()
    
    # Install requirements
    install_requirements()
    
    print("\nSetup completed successfully!")
    print("\nNext steps:")
    print("1. Activate your virtual environment")
    print("2. Run the demo: python demo.py")
    print("3. Press 'q' to quit the demo")

if __name__ == "__main__":
    main() 
"""
Build script for creating standalone Tracking Analysis executable.

Usage:
    python build_standalone.py

Requirements:
    pip install pyinstaller

Output:
    dist/TrackingAnalysis/  - Folder with executable and dependencies
"""

import subprocess
import sys
import shutil
from pathlib import Path

def main():
    project_root = Path(__file__).parent
    
    print("=" * 60)
    print("  Building Tracking Analysis Standalone")
    print("=" * 60)
    
    # Check PyInstaller is installed
    try:
        import PyInstaller
        print(f"\n[OK] PyInstaller version: {PyInstaller.__version__}")
    except ImportError:
        print("\n[!] PyInstaller not found. Installing...")
        subprocess.run([sys.executable, "-m", "pip", "install", "pyinstaller"], check=True)
    
    # Clean previous builds
    print("\n[1/4] Cleaning previous builds...")
    for folder in ['build', 'dist']:
        path = project_root / folder
        if path.exists():
            shutil.rmtree(path)
            print(f"  Removed {folder}/")
    
    # Run PyInstaller
    print("\n[2/4] Running PyInstaller (this may take several minutes)...")
    print("      Please wait...\n")
    spec_file = project_root / 'tracking_analysis.spec'
    
    result = subprocess.run(
        [sys.executable, "-m", "PyInstaller", str(spec_file), "--noconfirm"],
        cwd=str(project_root)
    )
    
    if result.returncode != 0:
        print("\n[ERROR] Build failed!")
        print("  Check the output above for errors.")
        print("  Common issues:")
        print("    - Missing dependencies: pip install <package>")
        print("    - Antivirus blocking: Add exception for project folder")
        sys.exit(1)
    
    # Create results folder in dist
    print("\n[3/4] Creating default folders...")
    dist_path = project_root / 'dist' / 'TrackingAnalysis'
    
    if not dist_path.exists():
        print("[ERROR] Build output not found!")
        sys.exit(1)
    
    (dist_path / 'results').mkdir(exist_ok=True)
    (dist_path / 'results' / 'states').mkdir(exist_ok=True)
    print("  Created results/ folder")
    print("  Created results/states/ folder")
    
    # Create README for distribution
    print("\n[4/4] Creating distribution README...")
    readme_content = """# Tracking Analysis - Standalone Application

## How to Use

1. **Run the application:**
   - Double-click `TrackingAnalysis.exe`
   - A console window will open showing the server status
   - Your browser will automatically open to the application

2. **Load your data:**
   - In the app, go to "Load New Data"
   - Select your data folder containing CSV files
   - Files should follow the naming convention:
     `Participant_XXXX_Tracking_blob_experiment_XXarcmin_vX_condition.csv`

3. **Close the application:**
   - Close the browser tab
   - Close the console window (or press Ctrl+C)

## Folder Structure

```
TrackingAnalysis/
|-- TrackingAnalysis.exe    # Main executable
|-- results/                # Output folder for analysis results
|   |-- states/             # Saved analysis states
|-- [other files]           # Dependencies (do not modify)
```

## System Requirements

- Windows 10 or later
- Modern web browser (Chrome, Firefox, Edge)
- ~500MB disk space
- No internet connection required

## Troubleshooting

**App doesn't start:**
- Make sure no other application is using port 8501
- Try running as Administrator
- Check if antivirus is blocking the application

**Browser doesn't open:**
- Manually open http://localhost:8501 in your browser

**Data won't load:**
- Check that CSV files have the correct format (Frame, Target_X, Target_Y, Mouse_X, Mouse_Y)
- Ensure files follow the naming convention

**Performance issues:**
- Close other resource-intensive applications
- Use a smaller dataset for initial testing

## Data Format

Your CSV files should have these columns:
- Frame: Frame number (0-indexed)
- Target_X: Target X position in pixels
- Target_Y: Target Y position in pixels  
- Mouse_X: Mouse/response X position in pixels
- Mouse_Y: Mouse/response Y position in pixels

## Support

For issues or questions, contact the Psychophysics Research Group.

---
Built with Tracking Analysis v1.0
"""
    
    with open(dist_path / 'README.txt', 'w', encoding='utf-8') as f:
        f.write(readme_content)
    
    # Calculate size
    total_size = sum(f.stat().st_size for f in dist_path.rglob('*') if f.is_file())
    size_mb = total_size / (1024 * 1024)
    
    print("\n" + "=" * 60)
    print("  BUILD COMPLETE!")
    print("=" * 60)
    print(f"\n  Output location: {dist_path}")
    print(f"  Total size: {size_mb:.1f} MB")
    print(f"\n  To distribute:")
    print(f"    1. Zip the entire 'TrackingAnalysis' folder")
    print(f"    2. Share the zip file (~{size_mb/3:.0f}-{size_mb/2:.0f} MB compressed)")
    print(f"    3. Recipients unzip and run TrackingAnalysis.exe")
    print("\n" + "=" * 60)

if __name__ == "__main__":
    main()

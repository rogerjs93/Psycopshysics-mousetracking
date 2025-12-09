"""
Tracking Analysis Launcher
==========================
Standalone launcher for the Tracking Analysis Streamlit application.
This script starts a local Streamlit server and opens the browser.
"""

import sys
import os
import socket
import webbrowser
import threading
import time
import traceback
import logging
from pathlib import Path
from datetime import datetime

# Set up logging to file
def setup_logging(base_path):
    """Set up logging to capture errors."""
    log_dir = base_path / 'logs'
    log_dir.mkdir(exist_ok=True)
    log_file = log_dir / f'launcher_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
    
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return log_file

def find_free_port(start_port=8501, max_attempts=100):
    """Find a free port starting from start_port."""
    for port in range(start_port, start_port + max_attempts):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('localhost', port))
                return port
        except OSError:
            continue
    raise RuntimeError(f"Could not find free port in range {start_port}-{start_port + max_attempts}")

def open_browser(port, delay=3):
    """Open browser after a delay."""
    time.sleep(delay)
    webbrowser.open(f'http://localhost:{port}')

def main():
    """Main entry point for the standalone application."""
    # Determine base path first
    if getattr(sys, 'frozen', False):
        # Running as compiled executable
        exe_dir = Path(sys.executable).parent
        base_path = Path(sys._MEIPASS)
        app_path = base_path / 'tracking_analysis' / 'ui' / 'app.py'
        log_base = exe_dir  # Put logs next to exe
    else:
        # Running as script
        exe_dir = Path(__file__).parent
        base_path = Path(__file__).parent
        app_path = base_path / 'tracking_analysis' / 'ui' / 'app.py'
        log_base = base_path
    
    # Set up logging
    log_file = setup_logging(log_base)
    logger = logging.getLogger(__name__)
    
    try:
        logger.info("=" * 60)
        logger.info("  Tracking Analysis - Standalone Application Starting")
        logger.info("=" * 60)
        logger.info(f"  Python version: {sys.version}")
        logger.info(f"  Frozen: {getattr(sys, 'frozen', False)}")
        logger.info(f"  Executable: {sys.executable}")
        logger.info(f"  Base path: {base_path}")
        logger.info(f"  App path: {app_path}")
        logger.info(f"  App exists: {app_path.exists()}")
        logger.info(f"  Log file: {log_file}")
        
        # Check if app file exists
        if not app_path.exists():
            logger.error(f"App file not found: {app_path}")
            # Try to list what's in the directory
            tracking_dir = base_path / 'tracking_analysis'
            if tracking_dir.exists():
                logger.info(f"Contents of {tracking_dir}:")
                for item in tracking_dir.iterdir():
                    logger.info(f"  - {item.name}")
                ui_dir = tracking_dir / 'ui'
                if ui_dir.exists():
                    logger.info(f"Contents of {ui_dir}:")
                    for item in ui_dir.iterdir():
                        logger.info(f"  - {item.name}")
            input("\nPress Enter to exit...")
            return 1
        
        # Set environment variables to prevent issues
        os.environ['STREAMLIT_BROWSER_GATHER_USAGE_STATS'] = 'false'
        os.environ['STREAMLIT_SERVER_HEADLESS'] = 'true'
        
        # Find free port
        logger.info("Finding free port...")
        port = find_free_port()
        logger.info(f"Using port: {port}")
        
        print("\n" + "=" * 60)
        print("  Tracking Analysis - Standalone Application")
        print("=" * 60)
        print(f"\n  Starting server on port {port}...")
        print(f"  App will open in your browser automatically.")
        print(f"\n  Log file: {log_file}")
        print("\n  To stop the server, close this window or press Ctrl+C")
        print("=" * 60 + "\n")
        
        # Open browser in background thread
        logger.info("Starting browser thread...")
        browser_thread = threading.Thread(target=open_browser, args=(port,))
        browser_thread.daemon = True
        browser_thread.start()
        
        # Import Streamlit
        logger.info("Importing streamlit...")
        try:
            from streamlit.web import cli as stcli
            logger.info("Streamlit imported successfully")
        except Exception as e:
            logger.error(f"Failed to import streamlit: {e}")
            logger.error(traceback.format_exc())
            input("\nPress Enter to exit...")
            return 1
        
        # Configure sys.argv for Streamlit
        # Note: global.developmentMode must be false for server.port to work
        sys.argv = [
            "streamlit", "run",
            str(app_path),
            "--global.developmentMode", "false",
            "--server.port", str(port),
            "--server.headless", "true",
            "--browser.gatherUsageStats", "false",
            "--theme.base", "light",
            "--server.maxUploadSize", "200",
        ]
        
        logger.info(f"Starting Streamlit with args: {sys.argv}")
        
        # Run Streamlit
        result = stcli.main()
        logger.info(f"Streamlit exited with code: {result}")
        return result
        
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        logger.error(traceback.format_exc())
        print(f"\n\nERROR: {e}")
        print(f"\nSee log file for details: {log_file}")
        input("\nPress Enter to exit...")
        return 1

if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        print(f"\n\nFATAL ERROR: {e}")
        print(traceback.format_exc())
        input("\nPress Enter to exit...")

"""
CreditLens — Root Application Delegate
======================================
Serves as the root entry point for HuggingFace Spaces, Docker, and local development.
Delegates cleanly to the unified production server in server/app.py.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).parent))

from server.app import main

if __name__ == "__main__":
    main()

"""Make the repository root importable when pytest runs from anywhere."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

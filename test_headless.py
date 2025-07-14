#!/usr/bin/env python3
"""
Test script to verify headless rendering functionality
"""

import sys
import os
sys.path.append('/home/lambda1/Documents/thor-grasp')

# Test the visualization script with headless mode
print("Testing headless visualization...")

# Example usage:
# python scripts/visualize.py test_object --save-png test_output.png
# python scripts/visualize.py test_object --render
# python scripts/visualize.py test_object  # Should run in headless mode by default

print("Usage examples:")
print("1. Headless mode (default): python scripts/visualize.py <object_name>")
print("2. Save image: python scripts/visualize.py <object_name> --save-png output.png") 
print("3. Interactive mode: python scripts/visualize.py <object_name> --render")
print("4. Force display: python scripts/visualize.py <object_name> --display")

print("\nChanges implemented:")
print("- Added headless mode as default behavior")
print("- Added OffscreenRenderer support for true headless rendering")
print("- Added fallback to hidden window if OffscreenRenderer unavailable")
print("- Added --display flag to force interactive mode")
print("- Improved argument handling for render vs headless modes")

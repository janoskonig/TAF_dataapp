#!/usr/bin/env python3
"""
Quick script to check for statistically significant results.
Run this from the Flask app directory after activating your virtual environment.
"""

print("To check for statistically significant results:")
print("\n1. Make sure your Flask app is running:")
print("   python3 main.py")
print("\n2. Visit the results page in your browser:")
print("   http://localhost:5000/results")
print("\n3. Look for the 'V. Keresztmetszeti vizsgálat' section")
print("   - Results with p < 0.05 will be highlighted in RED")
print("   - Results with p ≥ 0.05 will be in BLUE")
print("\nAlternatively, you can check the results page programmatically.")
print("The analyses run automatically when you visit /results")



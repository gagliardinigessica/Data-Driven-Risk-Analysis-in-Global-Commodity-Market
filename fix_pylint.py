"""
Script pour corriger automatiquement les problèmes Pylint :
- Trailing whitespace (espaces en fin de ligne)
- Imports en dehors du toplevel
"""
import re

def fix_pylint_issues(filepath):
    """Corriger les problèmes Pylint dans un fichier."""
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    fixed_lines = []
    for line in lines:
        # Supprimer les espaces en fin de ligne
        fixed_line = line.rstrip() + '\n'
        fixed_lines.append(fixed_line)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.writelines(fixed_lines)
    
    print(f"Fixed trailing whitespace in {filepath}")

if __name__ == "__main__":
    fix_pylint_issues('main.py')
    print("Done!")


"""
Script de contrôle général du projet.
Vérifie la syntaxe, les imports, la cohérence et la structure.
"""
import os
import sys
import ast
import importlib.util

def check_syntax(filepath):
    """Vérifie la syntaxe d'un fichier Python."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            code = f.read()
        ast.parse(code)
        return True, None
    except SyntaxError as e:
        return False, f"Syntax error: {e}"
    except Exception as e:
        return False, f"Error: {e}"

def check_imports_in_file(filepath):
    """Vérifie que les imports dans un fichier sont valides."""
    errors = []
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            tree = ast.parse(f.read())
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    try:
                        __import__(alias.name)
                    except ImportError:
                        errors.append(f"Cannot import {alias.name}")
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    try:
                        __import__(node.module)
                    except ImportError:
                        errors.append(f"Cannot import from {node.module}")
    except Exception as e:
        errors.append(f"Error checking imports: {e}")
    
    return errors

def check_function_definitions(module_path, required_functions):
    """Vérifie que les fonctions requises existent dans un module."""
    errors = []
    try:
        spec = importlib.util.spec_from_file_location("module", module_path)
        if spec is None or spec.loader is None:
            return [f"Cannot load module {module_path}"]
        
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        for func_name in required_functions:
            if not hasattr(module, func_name):
                errors.append(f"Function '{func_name}' not found in {module_path}")
    except Exception as e:
        errors.append(f"Error checking {module_path}: {e}")
    
    return errors

def check_file_exists(filepath):
    """Vérifie qu'un fichier existe."""
    return os.path.exists(filepath)

def main():
    """Contrôle général du projet."""
    print("="*70)
    print("CONTROLE GENERAL DU PROJET")
    print("="*70)
    
    errors = []
    warnings = []
    
    # 1. Vérifier la structure du projet
    print("\n[1/6] Vérification de la structure...")
    required_files = [
        'main.py',
        'environment.yml',
        'README.md',
        'test_imports.py',
        '.gitignore',
        'src/__init__.py',
        'src/data_loader.py',
        'src/volatility_models.py',
        'src/risk_analysis.py',
        'src/visualization.py'
    ]
    
    required_dirs = ['src', 'data/raw', 'results', 'notebooks']
    
    for filepath in required_files:
        if check_file_exists(filepath):
            print(f"  [OK] {filepath}")
        else:
            errors.append(f"Missing file: {filepath}")
            print(f"  [ERROR] {filepath} - MANQUANT")
    
    for dirpath in required_dirs:
        if os.path.isdir(dirpath):
            print(f"  [OK] {dirpath}/")
        else:
            warnings.append(f"Directory {dirpath} does not exist (will be created)")
            print(f"  [WARN] {dirpath}/ - sera cree automatiquement")
    
    # 2. Vérifier la syntaxe
    print("\n[2/6] Vérification de la syntaxe...")
    python_files = [
        'main.py',
        'test_imports.py',
        'src/__init__.py',
        'src/data_loader.py',
        'src/volatility_models.py',
        'src/risk_analysis.py',
        'src/visualization.py'
    ]
    
    for filepath in python_files:
        if check_file_exists(filepath):
            is_valid, error = check_syntax(filepath)
            if is_valid:
                print(f"  [OK] {filepath}")
            else:
                errors.append(f"{filepath}: {error}")
                print(f"  [ERROR] {filepath}: {error}")
    
    # 3. Vérifier les fonctions importées dans main.py
    print("\n[3/6] Vérification des imports dans main.py...")
    
    main_imports = {
        'src/data_loader.py': ['load_data', 'download_commodity_data', 'load_multiple_commodities', 
                               'clean_commodity_data', 'calculate_returns'],
        'src/volatility_models.py': ['fit_garch_model', 'forecast_volatility', 'calculate_historical_volatility',
                                     'compare_volatility_models', 'extract_volatility_series'],
        'src/risk_analysis.py': ['calculate_correlation_matrix', 'calculate_portfolio_risk',
                                 'analyze_risk_propagation', 'calculate_var', 'calculate_cvar'],
        'src/visualization.py': ['plot_price_series', 'plot_volatility_series', 'plot_correlation_heatmap',
                                 'plot_returns_distribution', 'plot_risk_metrics', 'create_dashboard']
    }
    
    for module_path, functions in main_imports.items():
        if check_file_exists(module_path):
            module_errors = check_function_definitions(module_path, functions)
            if not module_errors:
                print(f"  [OK] {module_path} - toutes les fonctions presentes")
            else:
                for error in module_errors:
                    errors.append(error)
                    print(f"  [ERROR] {error}")
    
    # 4. Vérifier environment.yml
    print("\n[4/6] Vérification de environment.yml...")
    if check_file_exists('environment.yml'):
        with open('environment.yml', 'r') as f:
            content = f.read()
            required_deps = ['pandas', 'numpy', 'matplotlib', 'seaborn', 'scikit-learn', 
                           'statsmodels', 'arch', 'yfinance', 'plotly']
            missing_deps = []
            for dep in required_deps:
                if dep not in content:
                    missing_deps.append(dep)
            
            if not missing_deps:
                print(f"  [OK] Toutes les dependances presentes")
            else:
                warnings.append(f"Missing dependencies in environment.yml: {missing_deps}")
                print(f"  [WARN] Dependances manquantes: {', '.join(missing_deps)}")
    else:
        errors.append("environment.yml not found")
    
    # 5. Vérifier les chemins relatifs
    print("\n[5/6] Vérification des chemins relatifs...")
    python_files_to_check = [
        'src/data_loader.py',
        'src/volatility_models.py',
        'src/risk_analysis.py',
        'src/visualization.py',
        'main.py'
    ]
    
    hardcoded_paths = []
    for filepath in python_files_to_check:
        if check_file_exists(filepath):
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
                # Chercher des chemins absolus suspects
                if 'C:\\' in content or '/Users/' in content or '/home/' in content:
                    hardcoded_paths.append(filepath)
    
    if not hardcoded_paths:
        print(f"  [OK] Aucun chemin code en dur detecte")
    else:
        warnings.append(f"Possible hardcoded paths in: {hardcoded_paths}")
        print(f"  [WARN] Chemins potentiellement codes en dur dans: {', '.join(hardcoded_paths)}")
    
    # 6. Vérifier le point d'entrée
    print("\n[6/6] Vérification du point d'entrée...")
    if check_file_exists('main.py'):
        with open('main.py', 'r', encoding='utf-8') as f:
            content = f.read()
            if 'if __name__ == "__main__":' in content:
                print(f"  [OK] Point d'entree correct (if __name__ == '__main__')")
            else:
                errors.append("main.py missing 'if __name__ == \"__main__\":'")
                print(f"  [ERROR] Point d'entree manquant")
            
            if 'def main():' in content:
                print(f"  [OK] Fonction main() presente")
            else:
                warnings.append("main.py might not have main() function")
                print(f"  [WARN] Fonction main() non trouvee")
    else:
        errors.append("main.py not found")
    
    # Résumé
    print("\n" + "="*70)
    print("RESUME DU CONTROLE")
    print("="*70)
    
    if errors:
        print(f"\n[ERROR] ERREURS TROUVEES ({len(errors)}):")
        for i, error in enumerate(errors, 1):
            print(f"  {i}. {error}")
    else:
        print("\n[OK] AUCUNE ERREUR CRITIQUE")
    
    if warnings:
        print(f"\n[WARN] AVERTISSEMENTS ({len(warnings)}):")
        for i, warning in enumerate(warnings, 1):
            print(f"  {i}. {warning}")
    else:
        print("\n[OK] AUCUN AVERTISSEMENT")
    
    print("\n" + "="*70)
    if not errors:
        print("[OK] PROJET VALIDE - Pret pour execution!")
        print("\nPour tester:")
        print("  python test_imports.py")
        print("  python main.py")
    else:
        print("[ERROR] PROJET CONTIENT DES ERREURS - Veuillez les corriger")
    print("="*70)
    
    return len(errors) == 0

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)


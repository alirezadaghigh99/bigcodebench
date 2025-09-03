import ast
from itertools import combinations
from radon.metrics import mi_visit
from radon.complexity import cc_visit
import json
import os

class FunctionMetrics(ast.NodeVisitor):
    def __init__(self, global_vars, imported_modules):
        self.global_accesses = set()
        self.external_modules = set(imported_modules)
        self.global_vars = set(global_vars)

    def visit_Name(self, node):
        # track reads of module-level globals
        if isinstance(node.ctx, ast.Load) and node.id in self.global_vars:
            self.global_accesses.add(node.id)

    def visit_Attribute(self, node):
        # track any attribute access off a Name (e.g. sys.stdin → sys)
        if isinstance(node.value, ast.Name):
            self.external_modules.add(node.value.id)
        self.generic_visit(node)

def analyze_module(source: str):
    """
    Returns a dict with:
    MI : maintainability index for entire module
    LCOM1 : module-level cohesion (lack-of-cohesion) over functions
    CBO : module-level coupling (distinct external modules)
    CC : dict of cyclomatic complexity per function
    """
    try:
        tree = ast.parse(source)
        mi = mi_visit(source, False)
        blocks = cc_visit(source)

        # 1) collect module-level globals & imports
        globals_ = set()
        imports_ = set()
        for node in tree.body:
            if isinstance(node, ast.Assign):
                for tgt in node.targets:
                    if isinstance(tgt, ast.Name):
                        globals_.add(tgt.id)
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    imports_.add(alias.name.split('.')[0])
            elif isinstance(node, ast.ImportFrom) and node.module:
                imports_.add(node.module.split('.')[0])

        # 2) analyze each top-level function
        func_metrics = {}
        for node in tree.body:
            if isinstance(node, ast.FunctionDef):
                fm = FunctionMetrics(globals_, imports_)
                fm.visit(node)
                # cyclomatic for this function
                cc = next((b.complexity for b in blocks if b.name == node.name), 1)
                func_metrics[node.name] = {
                    'CC': cc,
                    'globals_used': fm.global_accesses,
                    'externals': fm.external_modules - {node.name}
                }

        # 3) module cohesion: LCOM1 over functions sharing globals
        names = list(func_metrics)
        P = Q = 0
        for f1, f2 in combinations(names, 2):
            if func_metrics[f1]['globals_used'] & func_metrics[f2]['globals_used']:
                Q += 1
            else:
                P += 1
        module_lcom1 = max(0, P - Q)

        # 4) module coupling: size of union of all externals
        module_cbo = len(set().union(*(m['externals'] for m in func_metrics.values())))

        return {
            'MI': mi,
            'LCOM1': module_lcom1,
            'CBO': module_cbo,
            'CC': {f: m['CC'] for f, m in func_metrics.items()},
            'avg_CC': sum(m['CC'] for m in func_metrics.values()) / len(func_metrics) if func_metrics else 0
        }
    except Exception as e:
        # Return default values for unparseable code
        return {
            'MI': 0,
            'LCOM1': 999, # high value indicates poor cohesion
            'CBO': 999, # high value indicates high coupling
            'CC': {},
            'avg_CC': 999 # high complexity
        }

def calculate_quality_score(metrics):
    """
    Calculate a single quality score from metrics.
    Higher score = better quality
    """
    # Normalize metrics to 0-100 scale and weight them
    mi_score = max(0, min(100, metrics['MI'])) # MI is already 0-100
    lcom_score = max(0, 100 - min(100, metrics['LCOM1'])) # Lower LCOM1 is better
    cbo_score = max(0, 100 - min(100, metrics['CBO'] * 10)) # Lower CBO is better
    cc_score = max(0, 100 - min(100, metrics['avg_CC'] * 10)) # Lower CC is better
    
    # Weight the scores (MI gets highest weight as it's comprehensive)
    quality_score = (mi_score * 0.4 + 
                    lcom_score * 0.2 + 
                    cbo_score * 0.2 + 
                    cc_score * 0.2)
    return quality_score

def load_code_from_generation(task_id, technique, model):
    """
    Load code from generation_output files
    """
    # Try different file naming patterns
    if technique == 'original':
        patterns = [
            f"generation_output/{model}/code_original_{model}_bigcodebench_output_llm1_v2.jsonl",
            f"generation_output/code_original_{model}_livecodebench_final.jsonl",
            f"generation_output/code_original_{model}_livecodebench.jsonl"
        ]
    else:
        patterns = [
            f"generation_output/{model}/code_{technique}_{model}_bigcodebench_output_llm1_v2.jsonl",
            f"generation_output/code_{technique}_{model}_livecodebench_final.jsonl",
            f"generation_output/code_{technique}_{model}_livecodebench.jsonl"
        ]
    
    for file_path in patterns:
        if os.path.exists(file_path):
            try:
                with open(file_path, 'r') as f:
                    for line in f:
                        data = json.loads(line)
                        if data.get('task_id') == task_id:
                            # Try different field names for the code
                            code_fields = ['response_code', 'code', 'solution', 'generated_code']
                            for field in code_fields:
                                if field in data and data[field]:
                                    return data[field]
                            return data.get('response_code', '')
            except (json.JSONDecodeError, IOError):
                continue
    return None

def compare_code_quality(task_id, technique_a, technique_b, model):
    """
    Compare code quality between two techniques
    Returns: technique_a, technique_b, or 'tie' based on quality metrics
    """
    code_a = load_code_from_generation(task_id, technique_a, model)
    code_b = load_code_from_generation(task_id, technique_b, model)
    
    if not code_a or not code_b:
        return 'tie' # Fallback if code can't be loaded
    
    metrics_a = analyze_module(code_a)
    metrics_b = analyze_module(code_b)
    
    score_a = calculate_quality_score(metrics_a)
    score_b = calculate_quality_score(metrics_b)
    
    # Determine winner based on quality scores
    score_diff = abs(score_a - score_b)
    if score_diff < 5: # Within 5 points is considered a tie
        return 'tie'
    elif score_a > score_b:
        return technique_a
    else:
        return technique_b

def compare_code_quality_detailed(task_id, technique_a, technique_b, model):
    """
    Compare code quality between two techniques with detailed metrics
    Returns: dict with winner, scores, and detailed metrics
    """
    code_a = load_code_from_generation(task_id, technique_a, model)
    code_b = load_code_from_generation(task_id, technique_b, model)
    
    if not code_a or not code_b:
        return {
            'winner': 'tie',
            'reason': 'code_not_found',
            'score_a': 0,
            'score_b': 0,
            'metrics_a': {},
            'metrics_b': {}
        }
    
    metrics_a = analyze_module(code_a)
    metrics_b = analyze_module(code_b)
    
    score_a = calculate_quality_score(metrics_a)
    score_b = calculate_quality_score(metrics_b)
    
    # Determine winner based on quality scores
    score_diff = abs(score_a - score_b)
    if score_diff < 5: # Within 5 points is considered a tie
        winner = 'tie'
        reason = 'scores_within_threshold'
    elif score_a > score_b:
        winner = technique_a
        reason = 'higher_quality_score'
    else:
        winner = technique_b
        reason = 'higher_quality_score'
    
    return {
        'winner': winner,
        'reason': reason,
        'score_a': score_a,
        'score_b': score_b,
        'score_diff': score_diff,
        'metrics_a': metrics_a,
        'metrics_b': metrics_b
    }
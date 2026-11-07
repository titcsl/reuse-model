import ast
import base64
import csv
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import re
import subprocess
import sys
import tokenize
import zlib

import pandas as pd

from restore_executed import recover


ROOT = Path(__file__).resolve().parent


def constants(source):
    result = {}
    for node in ast.parse(source).body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            if node.targets[0].id in ('CUDA_SOURCE', 'KERNEL_BUILD', 'PTX_ZLIB_BASE64'):
                result[node.targets[0].id] = ast.literal_eval(node.value)
    return result


class Comparable(ast.NodeTransformer):
    def visit_Expr(self, node):
        if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            return None
        return self.generic_visit(node)

    def visit_Assign(self, node):
        if isinstance(node.targets[0], ast.Name) and node.targets[0].id in ('CUDA_SOURCE', 'KERNEL_BUILD'):
            node.value = ast.Constant('checked separately')
        return self.generic_visit(node)


def normalized(source):
    return ast.dump(Comparable().visit(ast.parse(source)), include_attributes=False)


def main():
    csv_path = ROOT/'data/gpu_audit_v3_20261106_153155_70b1e6.csv'
    source, source_hash = recover(csv_path)
    readable_path = ROOT/'benchmark/gpu_audit_v3.py'
    readable = readable_path.read_text()
    provenance = json.loads((ROOT/'provenance/readable_source.json').read_text())
    assert source_hash == provenance['executed_source_sha256']
    assert hashlib.sha256(readable.encode()).hexdigest() == provenance['readable_source_sha256']
    assert normalized(source) == normalized(readable)
    original, cleaned = constants(source), constants(readable)
    original_kernel = re.sub(r'//[^\n]*', '', original['CUDA_SOURCE'])
    assert re.sub(r'\s+', '', original_kernel) == re.sub(r'\s+', '', cleaned['CUDA_SOURCE'])
    assert cleaned['CUDA_SOURCE'] == (ROOT/'benchmark/kernels.cu').read_text()
    assert hashlib.sha256(cleaned['CUDA_SOURCE'].encode()).hexdigest() == cleaned['KERNEL_BUILD']['source_sha256']
    expected_build = dict(original['KERNEL_BUILD'])
    expected_build['source_sha256'] = cleaned['KERNEL_BUILD']['source_sha256']
    assert expected_build == cleaned['KERNEL_BUILD']
    assert original['PTX_ZLIB_BASE64'] == cleaned['PTX_ZLIB_BASE64']
    ptx = zlib.decompress(base64.b64decode(cleaned['PTX_ZLIB_BASE64']))
    assert hashlib.sha256(ptx).hexdigest() == provenance['embedded_ptx_sha256']
    authored = list(ROOT.glob('*.py')) + list((ROOT/'benchmark').glob('*.py'))
    for path in authored:
        text = path.read_text()
        assert not [token for token in tokenize.generate_tokens(io.StringIO(text).readline) if token.type == tokenize.COMMENT], path
        assert not [node for node in ast.walk(ast.parse(text)) if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) and ast.get_docstring(node)], path
    assert '//' not in cleaned['CUDA_SOURCE'] and '/*' not in cleaned['CUDA_SOURCE']
    subprocess.run([sys.executable, str(readable_path), '--self-test'], check=True)
    spec = importlib.util.spec_from_file_location('readable_benchmark', readable_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    traces = 0
    with csv_path.open(newline='', encoding='utf-8-sig') as handle:
        for row in csv.DictReader(handle):
            if row['record_type'] != 'trial_plan':
                continue
            plan = json.loads(row['details'])
            expected = module.make_plan(int(row['session']))[int(row['trial'])-1]
            assert all(plan[key] == value for key, value in expected.items())
            if plan['paced']:
                warm, measured, hashes = module.trace(plan['scenario'], plan['seed'], plan['frames'], round(plan['warm_seconds']*60), plan['width'], plan['height'])
                assert hashes == [plan['warm_trace_sha256'], plan['state_trace_sha256']]
                traces += 1
    followup = pd.read_csv(ROOT/'results/followup_trials.csv')
    disagreements = []
    for key, group in followup[followup.phase == 'main'].groupby(['session', 'block']):
        baseline = group[group.policy == 'full_render'].iloc[0]
        for row in group[group.policy != 'full_render'].itertuples():
            counter = 100*(1-row.counter_j/baseline.counter_j)
            power = 100*(1-row.power_j/baseline.power_j)
            disagreements.append(counter*power < 0)
    assert len(disagreements) == 36 and sum(disagreements) == 10
    assert len(followup) == 144 and int(followup.frames.sum()) == 204297
    summary = json.loads((ROOT/'results/audit_summary.json').read_text())
    assert not summary['pilot']['problems'] and not summary['followup']['problems']
    report = {
        'status': 'passed', 'exact_source_checksum': source_hash,
        'python_structure_comparison': 'passed with documented constants and documentation normalized',
        'cuda_executable_token_comparison': 'passed', 'compiled_ptx_comparison': 'identical',
        'authored_python_files_without_comments_or_docstrings': len(authored),
        'regenerated_paced_trace_pairs': traces,
        'followup_plan_trials': 144, 'independently_recalculated_sign_disagreements': 10,
        'new_gpu_run_on_packaging_host': False,
        'scope': 'Host checks and reproducibility of the supplied acquisition, not external energy calibration'
    }
    (ROOT/'results/package_validation.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()

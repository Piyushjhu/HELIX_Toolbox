"""Reconcile logged shot filenames and local input availability before analysis.

Uses metadata-only file checks for cloud placeholders; never downloads data.
"""
import argparse
from collections import Counter, defaultdict
import json
import os
from pathlib import Path
import re

from helix_data_source import MixedParameters, trace_key
from helix_file_io import file_availability, read_parameter_table


def audit_inputs(settings, emit=print):
    root = Path(settings.get('input_dir') or '.').absolute()
    param_folder = settings.get('param_folder')
    explicit = settings.get('input_files')
    pattern = settings.get('input_pattern') or '*.csv'
    directories = [root]
    if settings.get('batch_mode'):
        directories = sorted(p for p in root.glob(settings.get('subfolder_pattern') or '*')
                             if p.is_dir() and (not param_folder or not (Path(param_folder).exists() and p.samefile(param_folder))))
    paths = ([Path(p).absolute() for p in explicit] if explicit and not settings.get('batch_mode') else
             [p.absolute() for d in directories for p in sorted(d.glob(pattern)) if p.is_file()])
    files = [{'path': str(p), 'availability': file_availability(p)} for p in paths]
    index = defaultdict(list)
    for item in files:
        index[trace_key(item['path'])].append(item)
    params = []
    expected = []
    blank_rows = 0
    mixed = MixedParameters()
    metadata_errors = []
    for path in sorted(Path(param_folder).iterdir()) if param_folder and Path(param_folder).is_dir() else []:
        if path.name.startswith('~$') or path.suffix.lower() not in ('.csv', '.xlsx', '.xls'):
            continue
        state = file_availability(path)
        params.append({'path': str(path), 'availability': state})
        if state != 'local':
            continue
        try:
            df = read_parameter_table(path)
            df.columns = [str(c).strip() for c in df.columns]
            is_mpdv = any(re.fullmatch(r'PDV_\d+_FileName', c) for c in df.columns)
            if is_mpdv:
                mixed.add_mpdv_frame(df, path.name, emit=lambda _: None)
                column = 'PDV_10_FileName'
            else:
                column = next((c for c in df.columns if re.sub(r'[^a-z0-9]', '', c.lower())
                               in ('pdvfilename', 'pdvfile', 'dvfilename', 'dvfile', 'filename', 'file')), None)
            if not column:
                raise ValueError('No recognized PDV filename column')
            for n, row in df.iterrows():
                key = trace_key(row[column])
                if not key:
                    blank_rows += 1
                    continue
                if settings.get('data_mode') == 'mpdv' and not is_mpdv:
                    continue
                expected.append({'filename': key, 'parameter_file': str(path), 'row': int(n) + 2,
                                 'probe': 'PDV_10' if is_mpdv else 'single_pdv'})
        except Exception as exc:
            metadata_errors.append({'path': str(path), 'error': str(exc)})
    complete = bool(params) and all(p['availability'] == 'local' for p in params) and not metadata_errors
    counts = Counter(row['filename'] for row in expected)
    for row in expected:
        matches = index.get(row['filename'], [])
        row['matches'] = [m['path'] for m in matches]
        row['status'] = ('duplicate_mapping' if counts[row['filename']] > 1 else
                         'missing' if not matches else 'ambiguous' if len(matches) > 1 else
                         matches[0]['availability'])
    expected_keys = set(counts)
    for item in files:
        key = trace_key(item['path'])
        item['role'] = ('expected' if key in expected_keys else
                        'noncentral_mpdv' if mixed.is_mpdv(key) and not mixed.allows(key) else
                        'unmapped' if complete else 'reconciliation_unavailable')
    relevant = [f for f in files if f['role'] != 'noncentral_mpdv']
    ready = (complete and bool(expected) and all(row['status'] == 'local' for row in expected)
             and all(f['availability'] == 'local' and f['role'] == 'expected' for f in relevant))
    report = {
        'input_dir': str(root), 'parameter_folder': param_folder,
        'reconciliation_complete': complete, 'ready': ready,
        'scope_note': 'Exact logged filenames across the supplied parameter folder versus the selected inputs. '
                      'Use matching parameter/input scope. Blank filename rows are not counted as acquired traces. '
                      'Local status checks filesystem flags/size, not waveform validity or a full-content read.',
        'summary': {'waveform_files': len(files),
                    'waveform_availability': dict(Counter(f['availability'] for f in files)),
                    'parameter_files': len(params),
                    'parameter_availability': dict(Counter(p['availability'] for p in params)),
                    'logged_nonblank_rows': len(expected), 'blank_filename_rows': blank_rows,
                    'expected_status': dict(Counter(row['status'] for row in expected)),
                    'file_roles': dict(Counter(f['role'] for f in files))},
        'parameters': params, 'metadata_errors': metadata_errors,
        'expected': expected, 'files': files,
    }
    emit(json.dumps(report['summary'], indent=2))
    emit('READY' if ready else 'NOT READY: do not treat analysis results as a complete dataset.')
    if not complete:
        emit('Expected-shot reconciliation is incomplete: parameter logs are unavailable or unreadable.')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, help='Master JSON/YAML configuration to audit')
    parser.add_argument('--report', required=True, help='Local JSON report path (outside OneDrive recommended)')
    args = parser.parse_args()
    require = file_availability(args.config)
    if require != 'local':
        parser.error(f'Config unavailable: {require}: {args.config}')
    with open(args.config) as stream:
        if args.config.lower().endswith(('.yml', '.yaml')):
            import yaml
            config = yaml.safe_load(stream)
        else:
            config = json.load(stream)
    report = audit_inputs(config['cli_settings'])
    target = Path(args.report).absolute()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2))
    print(f'Full report: {target}')
    return 0 if report['ready'] else 1


if __name__ == '__main__':
    raise SystemExit(main())

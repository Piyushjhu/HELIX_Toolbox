"""MPDV input adaptation; the legacy single-PDV loaders remain unchanged."""

import math
import os

import pandas as pd


MPDV_NOTE = (
    "MPDV central-probe mode: processing PDV_10 only. "
    "All other probes' data are suppressed for this MPDV run."
)
_OUTPUT_SUFFIXES = (
    '--vel-smooth-with-uncert', '--velocity--smooth', '--velocity--uncert',
    '--velocity', '--noise--frac', '--results',
)


def validate_data_mode(mode):
    if mode not in ('single_pdv', 'mpdv'):
        raise ValueError("data_mode must be 'single_pdv' or 'mpdv'")
    return mode


def trace_key(value):
    """Exact acquisition basename, preserving channel, timestamp and shot ID."""
    if value is None or pd.isna(value):
        return ''
    name = str(value).strip().strip('\"\'').replace('\\', '/').rsplit('/', 1)[-1]
    if name.lower() in ('', 'nan', 'none', '0'):
        return ''
    for ext in ('.csv', '.txt', '.dat', '.trc'):
        if name.lower().endswith(ext):
            name = name[:-len(ext)]
            break
    for suffix in _OUTPUT_SUFFIXES:
        if name.endswith(suffix):
            return name[:-len(suffix)]
    return name


def load_mpdv_parameters(folder, emit=print):
    """Read explicit PDV_10 mappings from CSV/Excel without guessing a channel.

    All parameter files are considered: batch parent directory names need not
    match log names. Duplicate acquisition names are errors, never last-row wins.
    """
    if not folder or not os.path.isdir(folder):
        raise ValueError('MPDV requires a valid parameter folder with PDV_10_FileName.')
    result = {}
    sources = {}
    matched_files = 0
    skipped = 0
    for name in sorted(os.listdir(folder)):
        if name.startswith('~$') or not name.lower().endswith(('.csv', '.xlsx', '.xls')):
            continue
        path = os.path.join(folder, name)
        df = pd.read_csv(path) if name.lower().endswith('.csv') else pd.read_excel(path)
        df.columns = [str(c).strip() for c in df.columns]
        if 'PDV_10_FileName' not in df.columns:
            emit(f'[MPDV] Ignoring {name}: no PDV_10_FileName column.')
            continue
        matched_files += 1
        for index, row in df.iterrows():
            key = trace_key(row['PDV_10_FileName'])
            if not key:
                skipped += 1
                continue
            source = f'{name}, row {index + 2}'
            if key in result:
                raise ValueError(f'Ambiguous PDV_10 filename {key}: {sources[key]} and {source}')
            info = {k: v for k, v in row.items() if not pd.isna(v)}
            for column, value in row.items():
                if column.startswith('PDV_10_'):
                    info['PDV_' + column[len('PDV_10_'):]] = value
            info['PDV_FileName'] = key
            info['data_mode'] = 'mpdv'
            info['selected_probe'] = 'PDV_10'
            wavelength = info.get('PDV_Target_Wavelength (m)')
            if wavelength is not None and not pd.isna(wavelength):
                wavelength = float(wavelength)
                if not math.isfinite(wavelength) or wavelength <= 0:
                    raise ValueError(f'Invalid PDV_10 target wavelength in {source}')
                info['PDV_Target_Wavelength (m)'] = wavelength
            result[key] = info
            sources[key] = source
    if not matched_files:
        raise ValueError('MPDV parameter folder has no PDV_10_FileName column.')
    emit(f'[MPDV] Loaded {len(result)} PDV_10 records; skipped {skipped} rows with blank central filenames.')
    return result


def select_central_files(files, parameters, require_match=True):
    """Filter both raw and derived files using exact logged acquisition names."""
    selected = []
    seen = set()
    for path in files or []:
        key = trace_key(path)
        if key not in parameters:
            continue
        if key in seen:
            raise ValueError(f'Ambiguous PDV_10 input: more than one file for {key}')
        seen.add(key)
        selected.append(path)
    if require_match and not selected:
        raise ValueError('No input files match PDV_10_FileName. Check the parameter folder and input selection.')
    return selected


def central_summary(df, keys):
    """Keep only explicitly mapped central rows when reading existing summaries."""
    for column in ('file_name', 'Filename', 'filename', 'PDV_FileName'):
        if column in df.columns:
            return df.loc[df[column].map(trace_key).isin(keys)].copy()
    raise ValueError('Cannot identify PDV_10 rows: summary has no filename column.')


def mpdv_header_lines(path, configured):
    """Recognize the supplied LeCroy CSV layout; retain configured fallback."""
    with open(path, encoding='utf-8-sig') as stream:
        for index, line in zip(range(100), stream):
            if [part.strip().lower() for part in line.split(',')] == ['time', 'ampl']:
                return index  # pandas consumes Time,Ampl as its column header
    return configured

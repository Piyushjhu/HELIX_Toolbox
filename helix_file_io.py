"""Check local availability without opening cloud placeholders."""
import os
import queue
import stat
import threading


class InputUnavailableError(ValueError):
    """A required input cannot be read safely; the run is incomplete."""


def file_availability(path):
    try:
        info = os.stat(path)
    except FileNotFoundError:
        return 'missing'
    except OSError:
        return 'inaccessible'
    if not stat.S_ISREG(info.st_mode):
        return 'not_a_file'
    # SF_DATALESS from macOS sys/stat.h (absent in older Python stat modules).
    if getattr(info, 'st_flags', 0) & getattr(stat, 'SF_DATALESS', 0x40000000):
        return 'cloud_only'
    # Windows FILE_ATTRIBUTE_OFFLINE / FILE_ATTRIBUTE_RECALL_ON_DATA_ACCESS.
    if getattr(info, 'st_file_attributes', 0) & (0x1000 | 0x400000):
        return 'cloud_only'
    if info.st_size == 0:
        return 'empty'
    return 'local'


def require_local_files(paths):
    problems = [(os.fspath(p), file_availability(p)) for p in paths]
    problems = [(p, status) for p, status in problems if status != 'local']
    if problems:
        details = '\n'.join(f'  {status}: {p}' for p, status in problems)
        raise InputUnavailableError(
            f'Input availability check failed for {len(problems)} file(s). '
            'No unavailable input will be treated as a rejected shot.\n' + details +
            '\nFor OneDrive, mark the dataset folder Always Keep on This Device, '
            'wait for downloading/sync to finish, then rerun. '
            'Check missing files against the acquisition log/source backup.')


def read_parameter_table(path, timeout=30):
    """Fail visibly on a timed-out read, without waiting forever at pool exit.

    A daemon worker may remain blocked in OS I/O after the timeout. It owns its
    path and cannot mutate the caller's metadata; it cannot prevent CLI exit.
    """
    require_local_files([path])
    result = queue.Queue(maxsize=1)

    def read():
        try:
            import pandas as pd
            table = pd.read_csv(path) if str(path).lower().endswith('.csv') else pd.read_excel(path)
            result.put((table, None))
        except Exception as exc:
            result.put((None, exc))

    threading.Thread(target=read, daemon=True).start()
    try:
        table, error = result.get(timeout=timeout)
    except queue.Empty as exc:
        raise InputUnavailableError(f'Parameter read timed out after {timeout}s: {path}. '
                                    'Run stopped; metadata was not skipped.') from exc
    if error is not None:
        raise InputUnavailableError(f'Cannot read parameter file {path}: {error}. '
                                    'Run stopped; metadata was not skipped.') from error
    return table

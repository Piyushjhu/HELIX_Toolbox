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


def _stage_file(source, destination):
    """Worker process: reading forces the cloud provider to materialize content."""
    import hashlib
    import json
    import tempfile
    from pathlib import Path
    source = Path(source)
    destination = Path(destination)
    before = source.stat()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        digest = hashlib.sha256()
        size = 0
        with source.open('rb') as src, tempfile.NamedTemporaryFile(
                dir=destination.parent, prefix=f'.{destination.name}.download-', delete=False) as dst:
            temporary = dst.name
            while True:
                chunk = src.read(1024 * 1024)
                if not chunk:
                    break
                dst.write(chunk)
                digest.update(chunk)
                size += len(chunk)
            dst.flush()
            os.fsync(dst.fileno())
        after = source.stat()
        if not size or size != before.st_size or (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise InputUnavailableError(f'Source empty, truncated, or changed during download: {source}')
        os.replace(temporary, destination)
        temporary = None
        return {'bytes': size, 'sha256': digest.hexdigest(), 'cached_path': str(destination)}
    finally:
        if temporary:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass


def download_and_verify(paths, cache_dir, timeout=120, attempts=3, workers=4, retry_delay=5, emit=print):
    """Request full reads in killable subprocesses; only return verified copies.

    Each source is reread on every invocation. An old cache entry never qualifies
    as a successful download. Failure manifests remain available for recovery.
    """
    from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
    from pathlib import Path
    import hashlib
    import json
    import shutil
    import subprocess
    import sys
    import time
    if timeout <= 0 or attempts < 1 or workers < 1 or retry_delay < 0:
        raise ValueError('Download timeout, attempts, and workers must be positive.')
    sources = list(dict.fromkeys(os.path.abspath(p) for p in paths))
    cache = Path(cache_dir).expanduser().resolve()
    # A cloud-synced cache would reintroduce the same I/O problem during analysis.
    if 'CloudStorage' in cache.parts or any(p.lower().startswith('onedrive') for p in cache.parts):
        raise ValueError('Input cache must be outside OneDrive/CloudStorage.')
    cache.mkdir(parents=True, exist_ok=True)
    records = {p: {'source': p, 'initial_status': file_availability(p), 'status': 'pending'} for p in sources}
    if sys.platform == 'darwin' and any(
            r['initial_status'] == 'cloud_only' and
            any(part.lower().startswith('onedrive') for part in Path(p).parts)
            for p, r in records.items()):
        emit('[Download] Requesting OneDrive sync app startup for cloud-only files.')
        try:
            started = subprocess.run(['open', '-a', 'OneDrive'], capture_output=True, text=True, timeout=10)
            if started.returncode:
                emit(f'[Download] OneDrive startup warning: {started.stderr.strip()}')
        except (OSError, subprocess.TimeoutExpired) as exc:
            emit(f'[Download] Cannot start OneDrive automatically: {exc}')
    required_bytes = sum(os.stat(p).st_size for p in sources if records[p]['initial_status'] in ('local', 'cloud_only'))
    if required_bytes > shutil.disk_usage(cache).free:
        raise InputUnavailableError(f'Not enough local cache space for {required_bytes} bytes: {cache}')
    worker_script = os.path.abspath(__file__)

    def prepare(path):
        record = records[path]
        state = record['initial_status']
        if state not in ('local', 'cloud_only'):
            record.update(status='failed', error=f'Input is {state}')
            return
        parent_key = hashlib.sha256(os.path.dirname(path).encode()).hexdigest()[:20]
        destination = cache / parent_key / os.path.basename(path)
        for attempt in range(1, attempts + 1):
            emit(f'[Download] {attempt}/{attempts}: {os.path.basename(path)} ({file_availability(path)})')
            record['attempts'] = attempt
            try:
                result = subprocess.run([sys.executable, worker_script, '--stage', path, str(destination)],
                                        capture_output=True, text=True, timeout=timeout)
                if result.returncode:
                    raise InputUnavailableError(result.stderr.strip() or 'Download worker failed')
                record.update(json.loads(result.stdout))
                record.pop('error', None)
                record.update(status='verified', source_status_after=file_availability(path))
                emit(f'[Download] Verified {os.path.basename(path)}: {record["bytes"]} bytes')
                return
            except (subprocess.TimeoutExpired, ValueError, OSError) as exc:
                if isinstance(exc, subprocess.TimeoutExpired):
                    for partial in destination.parent.glob(f'.{destination.name}.download-*'):
                        partial.unlink(missing_ok=True)
                record['error'] = f'{type(exc).__name__}: {exc}'
                emit(f'[Download] Attempt failed for {os.path.basename(path)}: {record["error"]}')
                if attempt < attempts and retry_delay:
                    time.sleep(retry_delay)
        record['status'] = 'failed'

    def save_manifest():
        report = {'files': [dict(r) for r in records.values()],
                  'ready': all(r['status'] == 'verified' for r in records.values())}
        temporary = cache / 'download_manifest.json.tmp'
        temporary.write_text(json.dumps(report, indent=2))
        os.replace(temporary, cache / 'download_manifest.json')

    save_manifest()
    try:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            pending = {pool.submit(prepare, p) for p in sources}
            last_update = time.monotonic()
            while pending:
                completed, pending = wait(pending, timeout=1, return_when=FIRST_COMPLETED)
                for future in completed:
                    future.result()
                if completed:
                    save_manifest()
                if time.monotonic() - last_update >= 10:
                    ready = sum(r['status'] == 'verified' for r in records.values())
                    emit(f'[Download status] {ready}/{len(sources)} verified; {len(pending)} pending. Analysis has not started.')
                    last_update = time.monotonic()
    finally:
        save_manifest()
    failed = [p for p, r in records.items() if r['status'] != 'verified']
    if failed:
        raise InputUnavailableError(f'Download verification failed for {len(failed)}/{len(sources)} inputs. '
                                    f'Analysis has not started. See {cache / "download_manifest.json"}')
    emit(f'[Download status] All {len(sources)} files fully read and verified in {cache}.')
    return {p: r['cached_path'] for p, r in records.items()}


if __name__ == '__main__':
    import json
    import sys
    if len(sys.argv) != 4 or sys.argv[1] != '--stage':
        raise SystemExit('Internal worker usage: --stage SOURCE DESTINATION')
    try:
        print(json.dumps(_stage_file(sys.argv[2], sys.argv[3])))
    except Exception as exc:
        print(f'{type(exc).__name__}: {exc}', file=sys.stderr)
        raise SystemExit(1)

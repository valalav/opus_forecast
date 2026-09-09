"""Immutable captures of calculation outputs and their locally verified inputs.

This is a capture journal, not proof that the latest public release was fetched.
Publication dates are unknown unless provided by the caller. Facts are joined in
a separate evaluation artifact; they never modify a prediction snapshot.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from uuid import uuid4


class ForecastJournalError(ValueError):
    """A snapshot failed validation and was not published."""


def _utc_now():
    return datetime.now(timezone.utc)


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2,
                       allow_nan=False) + '\n').encode('utf-8')


def _contained_path(root, name):
    if not isinstance(name, str) or Path(name).is_absolute() or '..' in Path(name).parts:
        raise ForecastJournalError(f'Invalid source path: {name!r}')
    path = (root / name).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ForecastJournalError(f'Source escapes repository: {name}')
    return path


def _verify_inputs(source_root, expected):
    if not isinstance(expected, dict) or not expected:
        raise ForecastJournalError('A non-empty calculation source_manifest is required')
    for name, digest in expected.items():
        path = _contained_path(source_root / 'data', name)
        if digest is not None and not re.fullmatch(r'[a-f0-9]{64}', str(digest)):
            raise ForecastJournalError(f'Invalid SHA256: {name}')
        actual = _sha256(path) if path.is_file() else None
        if actual != digest:
            raise ForecastJournalError(f'Input changed or missing: {name}')


def forecast_code_manifest(source_root):
    """Capture source fingerprints before computation for later race detection."""
    root = Path(source_root).resolve()
    # Include uncommitted/new modules too: HEAD alone does not identify execution.
    paths = set()
    for directory in ('sirena', 'scripts'):
        paths.update(p for p in (root / directory).rglob('*.py') if p.is_file())
    for pattern in ('requirements*.txt', 'pyproject.toml', 'poetry.lock', 'uv.lock',
                    'setup.cfg', 'setup.py'):
        paths.update(root.glob(pattern))
    if not paths:
        raise ForecastJournalError('No application source files found under source_root')
    return {str(p.relative_to(root)): _sha256(p) for p in sorted(paths)}


def _git_metadata(root):
    def read(*args):
        result = subprocess.run(['git', '-C', str(root), *args], capture_output=True,
                                check=False, timeout=15)
        return result.stdout.decode('utf-8', errors='replace').strip() if result.returncode == 0 else None
    return {'head': read('rev-parse', 'HEAD'),
            'working_tree_status': read('status', '--porcelain', '--untracked-files=normal')}


def _store_object(path, expected, objects):
    """Verify copies, install without replacement, and deduplicate by content."""
    destination = objects / expected
    if destination.exists():
        if _sha256(destination) != expected:
            raise ForecastJournalError(f'Corrupted archived object: {expected}')
        return f'objects/sha256/{expected}'
    fd, name = tempfile.mkstemp(prefix='.object-', dir=objects)
    temporary = Path(name)
    try:
        with os.fdopen(fd, 'wb') as output, path.open('rb') as source:
            shutil.copyfileobj(source, output, length=1024 * 1024)
            output.flush()
            os.fsync(output.fileno())
        if _sha256(temporary) != expected:
            raise ForecastJournalError(f'Source changed during capture: {path}')
        try:
            os.link(temporary, destination)  # atomic create; never overwrite an object
        except FileExistsError:
            if _sha256(destination) != expected:
                raise ForecastJournalError(f'Corrupted archived object: {expected}')
        destination.chmod(0o444)
    finally:
        temporary.unlink(missing_ok=True)
    return f'objects/sha256/{expected}'


def _month(value):
    try:
        return datetime.fromisoformat(str(value).replace('Z', '+00:00')).date().replace(day=1)
    except (ValueError, TypeError) as exc:
        raise ForecastJournalError(f'Invalid observation/target date: {value!r}') from exc


def _reject_truth(value):
    # Full calculation JSON remains intact, but this API cannot attach facts.
    forbidden = {'actual', 'actuals', 'fact', 'facts', 'truth', 'future_truth', 'target_actuals'}
    if isinstance(value, dict):
        if forbidden.intersection(value):
            raise ForecastJournalError('Facts must be stored in a separate evaluation artifact')
        for item in value.values():
            _reject_truth(item)
    elif isinstance(value, list):
        for item in value:
            _reject_truth(item)


def _forecast_rows(results, recorded_at, code_id):
    last = _month(results['last_data_date'])
    capture_month = _month(recorded_at)
    dates = results.get('forecast_dates')
    forecasts = results.get('forecasts')
    if not isinstance(dates, list) or not dates or not isinstance(forecasts, dict) or not forecasts:
        raise ForecastJournalError('Non-empty forecast_dates and forecasts are required')
    targets = [_month(date) for date in dates]
    if targets != sorted(set(targets)) or any(target <= last for target in targets):
        raise ForecastJournalError('Targets must be unique, increasing, and after last observation')
    rows = []
    statuses = results.get('model_status', {})
    contract = results.get('input_contract', {})
    models = sorted(set(forecasts) | set(statuses))
    for model in models:
        values = forecasts.get(model)
        if values is not None and (not isinstance(values, list) or len(values) > len(dates)):
            raise ForecastJournalError(f'{model}: invalid forecast vector length/type')
        metadata = statuses.get(model, {})
        if not isinstance(metadata, dict):
            raise ForecastJournalError(f'{model}: invalid model_status')
        for i, target in enumerate(targets):
            value = values[i] if values is not None and i < len(values) else None
            if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float))
                                      or not math.isfinite(value)):
                raise ForecastJournalError(f'{model}: non-finite/non-numeric forecast')
            timing = ('pre_target_month' if target > capture_month else
                      'within_target_month' if target == capture_month else 'overdue_target')
            reason = metadata.get('reason')
            if value is None and not reason:
                reason = 'no_value_at_horizon' if values is not None else 'model_forecast_unavailable'
            rows.append({
                'model': model, 'model_version': metadata.get('model_version', code_id),
                'calculated_at': results.get('generated_at'), 'recorded_at': recorded_at,
                'calculated_at_timezone_note': 'Preserved from caller; a missing offset is unknown, not UTC',
                'origin_asof': recorded_at, 'asof_basis': 'local_input_capture_time',
                'last_observation': metadata.get('last_observation', results['last_data_date']),
                'target_month': target.strftime('%Y-%m'),
                'horizon_from_observation': (target.year-last.year)*12+target.month-last.month,
                'timing_class': timing,
                'prospective_before_target_month': timing == 'pre_target_month',
                'nowcast_stage': metadata.get('nowcast_stage'),
                'input_representation': metadata.get('input_representation', contract.get('representation')),
                'output_units': 'mom_percentage_points',
                'basket_vintage_id': metadata.get('basket_vintage_id'),
                'weight_vintage_id': metadata.get('weight_vintage_id'),
                'value': value, 'status': 'available' if value is not None else 'unavailable',
                'reason': reason, 'model_status': metadata,
                'forecast_nature': metadata.get('forecast_nature', 'model'),
                'input_evidence': 'captured_local_files',
                'historical_realtime_evidence_level': None,
                'target_fact_published_at': None,
                'target_fact_publication_status': 'unknown',
            })
    return rows


def publish_forecast_snapshot(results, journal_root, source_root):
    """Atomically publish a new snapshot; never change a previous run or fact.

    ``source_root`` is the repository root. The calculation's source_manifest
    contains paths relative to its data directory, with None for absent inputs.
    Inputs are checked against this manifest before AND after content capture.
    The caller must also validate the manifest across computation (precompute
    already does); this function cannot observe earlier reads by a model.

    The recorded as-of is the current capture time, never a backdated generated_at.
    Elapsed target months are explicitly overdue, irrespective of observation h.
    Source/code objects are shared immutable content-addressed files; snapshots
    retain references. Orphan objects after a failed capture are harmless and
    are not published runs. No public-release freshness claim is made here.
    """
    root = Path(source_root).resolve()
    journal = Path(journal_root).resolve()
    # Serializing first freezes caller-owned nested mappings and rejects NaN.
    try:
        payload = json.loads(_json_bytes(results))
    except (TypeError, ValueError) as exc:
        raise ForecastJournalError('Forecast payload is not finite JSON') from exc
    _reject_truth(payload)
    expected = payload.get('source_manifest')
    _verify_inputs(root, expected)
    code = forecast_code_manifest(root)
    calculation_code = payload.get('calculation_code_manifest')
    if calculation_code is not None and calculation_code != code:
        raise ForecastJournalError('Application code changed since calculation started')
    code_id = hashlib.sha256(_json_bytes(code)).hexdigest()
    recorded_at = _utc_now().isoformat()
    rows = _forecast_rows(payload, recorded_at, code_id)
    run_id = _utc_now().strftime('%Y%m%dT%H%M%S%fZ') + '_' + uuid4().hex
    journal.mkdir(parents=True, exist_ok=True)
    objects = journal / 'objects' / 'sha256'
    objects.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix='.staging-', dir=journal))
    destination = journal / run_id
    try:
        inputs = {}
        for name, digest in expected.items():
            obj = _store_object(_contained_path(root / 'data', name), digest, objects) if digest else None
            inputs[name] = {'sha256': digest, 'object_path': obj,
                            'capture_status': 'captured' if digest else 'absent',
                            'published_at': None, 'retrieved_at': None,
                            'captured_at': recorded_at,
                            'observation_period': None, 'vintage_id': digest,
                            'metadata_note': 'Publication/retrieval dates not inferred from file timestamps'}
        code_objects = {name: {'sha256': digest,
                              'object_path': _store_object(_contained_path(root, name), digest, objects)}
                        for name, digest in code.items()}
        packages = {}
        for package in ('numpy', 'pandas', 'scipy', 'scikit-learn', 'statsmodels', 'ngboost', 'prophet', 'interpret'):
            try:
                packages[package] = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                packages[package] = None
        manifest = {'schema_version': 1, 'run_id': run_id, 'recorded_at': recorded_at,
                    'source_root': str(root), 'inputs': inputs,
                    'artifact_sha256': {'forecast.json': hashlib.sha256(_json_bytes(payload)).hexdigest(),
                                        'forecast_rows.json': hashlib.sha256(_json_bytes(rows)).hexdigest()},
                    'code': {'version_sha256': code_id, 'files': code_objects,
                             'calculation_start_manifest_verified': calculation_code is not None,
                             **_git_metadata(root),
                             'scope': 'All Python files under sirena/scripts and root packaging/lock files'},
                    'environment': {'python': sys.version, 'platform': sys.platform, 'packages': packages},
                    'limits': ['Local captures do not prove latest external publication availability.',
                               'Unknown publication dates remain unknown; overdue targets are not before-target forecasts. Publication-based nowcast eligibility needs release evidence.',
                               'Source manifests define the covered inputs; undeclared model file reads are not covered.',
                               'Only listed application sources and package versions are archived; the runtime is not bundled.']}
        files = {'forecast.json': payload, 'forecast_rows.json': rows, 'manifest.json': manifest}
        for name, content in files.items():
            with (staging / name).open('xb') as stream:
                stream.write(_json_bytes(content))
                stream.flush()
                os.fsync(stream.fileno())
        _verify_inputs(root, expected)
        if code != forecast_code_manifest(root):
            raise ForecastJournalError('Application code changed during capture')
        for path in staging.iterdir():
            path.chmod(0o444)
        staging.rename(destination)
        return destination
    finally:
        if staging.exists():
            shutil.rmtree(staging)

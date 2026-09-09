import hashlib
import json
from datetime import datetime, timezone

import pytest

from sirena import forecast_journal as journal


@pytest.fixture
def capture(tmp_path, monkeypatch):
    root = tmp_path / 'repo'
    (root / 'data').mkdir(parents=True)
    (root / 'sirena').mkdir()
    (root / 'sirena' / 'model.py').write_text('VERSION = 1\n')
    source = root / 'data' / 'input.csv'
    source.write_text('month,cpi\n2026-07,100.2\n')
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    results = {'generated_at': '2026-09-09T09:00:00',
               'last_data_date': '2026-07-01', 'horizon': 3,
               'forecast_dates': ['2026-08-01', '2026-09-01', '2026-10-01'],
               'forecasts': {'Ridge': [0.2, 0.3, 0.4], 'Weekly': [0.1, None], 'Failed': None},
               'model_status': {'Failed': {'status': 'unavailable', 'reason': 'missing proxy'},
                                'Weekly': {'status': 'available', 'nowcast_stage': 'week_1'}},
               'input_contract': {'representation': 'raw'},
               'source_manifest': {'input.csv': digest, 'missing.csv': None},
               'calculation_code_manifest': journal.forecast_code_manifest(root)}
    monkeypatch.setattr(journal, '_utc_now', lambda: datetime(2026, 9, 9, 10, tzinfo=timezone.utc))
    return root, tmp_path / 'journal', results


def test_capture_preserves_payload_and_distinguishes_overdue_targets(capture):
    root, output, results = capture
    snapshot = journal.publish_forecast_snapshot(results, output, root)
    assert json.loads((snapshot / 'forecast.json').read_text()) == results
    rows = json.loads((snapshot / 'forecast_rows.json').read_text())
    ridge = [r for r in rows if r['model'] == 'Ridge']
    assert [r['timing_class'] for r in ridge] == ['overdue_target', 'within_target_month', 'pre_target_month']
    assert [r['horizon_from_observation'] for r in ridge] == [1, 2, 3]
    assert [r['prospective_before_target_month'] for r in ridge] == [False, False, True]
    assert ridge[0]['last_observation'] == '2026-07-01'
    assert ridge[0]['origin_asof'] == '2026-09-09T10:00:00+00:00'
    assert all(r['target_fact_publication_status'] == 'unknown' for r in rows)
    assert all(r['target_fact_published_at'] is None for r in rows)
    assert all(r['reason'] == 'missing proxy' for r in rows if r['model'] == 'Failed')
    assert [r['reason'] for r in rows if r['model'] == 'Weekly'] == [None, 'no_value_at_horizon', 'no_value_at_horizon']
    manifest = json.loads((snapshot / 'manifest.json').read_text())
    assert manifest['code']['calculation_start_manifest_verified'] is True
    assert manifest['inputs']['missing.csv']['capture_status'] == 'absent'
    assert manifest['inputs']['input.csv']['published_at'] is None
    for item in [*manifest['inputs'].values(), *manifest['code']['files'].values()]:
        if item['sha256']:
            obj = output / item['object_path']
            assert hashlib.sha256(obj.read_bytes()).hexdigest() == item['sha256']


def test_reruns_are_distinct_and_previous_snapshot_unchanged_and_objects_reused(capture):
    root, output, results = capture
    first = journal.publish_forecast_snapshot(results, output, root)
    before = {p.name: p.read_bytes() for p in first.iterdir()}
    objects = {p.name for p in (output / 'objects' / 'sha256').iterdir()}
    results['forecasts']['Ridge'][0] = 0.9
    second = journal.publish_forecast_snapshot(results, output, root)
    assert first != second
    assert before == {p.name: p.read_bytes() for p in first.iterdir()}
    assert objects == {p.name for p in (output / 'objects' / 'sha256').iterdir()}
    assert not list(output.glob('.staging-*'))


def test_stale_input_rejected_before_publication(capture):
    root, output, results = capture
    (root / 'data' / 'input.csv').write_text('changed')
    with pytest.raises(journal.ForecastJournalError, match='Input changed'):
        journal.publish_forecast_snapshot(results, output, root)
    assert not output.exists()


@pytest.mark.parametrize('change', ['input', 'missing_input_appears', 'code'])
def test_source_race_has_no_published_run(capture, monkeypatch, change):
    root, output, results = capture
    original = journal._store_object
    changed = False

    def store_and_mutate(*args):
        nonlocal changed
        result = original(*args)
        if not changed:
            changed = True
            if change == 'input':
                (root / 'data' / 'input.csv').write_text('changed during copy')
            elif change == 'missing_input_appears':
                (root / 'data' / 'missing.csv').write_text('new release')
            else:
                (root / 'sirena' / 'model.py').write_text('VERSION = 2')
        return result

    monkeypatch.setattr(journal, '_store_object', store_and_mutate)
    with pytest.raises(journal.ForecastJournalError, match='changed'):
        journal.publish_forecast_snapshot(results, output, root)
    assert [p.name for p in output.iterdir()] == ['objects']


def test_code_changed_since_computation_is_rejected(capture):
    root, output, results = capture
    (root / 'sirena' / 'model.py').write_text('VERSION = 2')
    with pytest.raises(journal.ForecastJournalError, match='since calculation started'):
        journal.publish_forecast_snapshot(results, output, root)
    assert not output.exists()


def test_backdated_generation_does_not_claim_prospective(capture):
    root, output, results = capture
    results['generated_at'] = '2026-07-20T00:00:00'
    snapshot = journal.publish_forecast_snapshot(results, output, root)
    rows = json.loads((snapshot / 'forecast_rows.json').read_text())
    assert all(r['timing_class'] == 'overdue_target' for r in rows if r['target_month'] == '2026-08')


@pytest.mark.parametrize('field', ['facts', 'future_truth', 'target_actuals'])
def test_cannot_attach_future_truth(capture, field):
    root, output, results = capture
    results[field] = {'2026-10': 0.4}
    with pytest.raises(journal.ForecastJournalError, match='separate evaluation'):
        journal.publish_forecast_snapshot(results, output, root)
    assert not output.exists()


def test_corrupt_reused_object_rejected(capture):
    root, output, results = capture
    first = journal.publish_forecast_snapshot(results, output, root)
    obj = output / 'objects' / 'sha256' / results['source_manifest']['input.csv']
    obj.chmod(0o644)
    obj.write_text('tampered')
    with pytest.raises(journal.ForecastJournalError, match='Corrupted archived object'):
        journal.publish_forecast_snapshot(results, output, root)
    assert sorted(p.name for p in output.iterdir()) == sorted(['objects', first.name])


@pytest.mark.parametrize('change', ['nan', 'duplicate_target', 'unsafe_source'])
def test_invalid_payload_rejected(capture, change):
    root, output, results = capture
    if change == 'nan':
        results['forecasts']['Ridge'][0] = float('nan')
    elif change == 'duplicate_target':
        results['forecast_dates'][1] = results['forecast_dates'][0]
    else:
        results['source_manifest']['../outside.csv'] = None
    with pytest.raises(journal.ForecastJournalError):
        journal.publish_forecast_snapshot(results, output, root)
    assert not output.exists()

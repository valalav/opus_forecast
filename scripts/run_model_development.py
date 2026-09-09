#!/usr/bin/env python3
"""Run a frozen protocol through the existing BacktestRunner."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.backtest_framework import BacktestRunner
from sirena.experiment_models import m1_adapters, M1_PARAMETERS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--protocol', required=True, type=Path)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text(encoding='utf-8'))
    run_id = protocol['run_id']
    if not run_id or Path(run_id).name != run_id or run_id in {'.', '..'}:
        raise ValueError('run_id must be one directory name')
    output = ROOT/'archive/results/model_development'/run_id
    runner = BacktestRunner(1, test_months=len(protocol['targets']), output_dir=str(output))
    adapters = m1_adapters()
    expected = {name: M1_PARAMETERS[name] for name in protocol['models']}
    if protocol['parameters'] != expected or protocol['seed'] != 42:
        raise ValueError('M1 adapter parameters/seed differ from locked protocol')
    selected = {name: adapters[name] for name in protocol['models']}
    runner.run_registered_experiment(protocol, selected)


if __name__ == '__main__':
    main()

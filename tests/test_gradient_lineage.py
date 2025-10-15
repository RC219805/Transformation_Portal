from __future__ import annotations
from __future__ import annotations

import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _trace_lineage(gradient: str) -> dict:
    script = f"""
const GradientLineage = require('./09_Client_Deliverables/Lantern_Logo_Implementation_Kit/gradient_lineage.js');
const lineage = new GradientLineage();
const result = lineage.trace('{gradient}');
process.stdout.write(JSON.stringify(result));
"""
    completed = subprocess.run(
        ['node', '-e', script],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(completed.stdout)


def test_brand_primary_lineage() -> None:
    data = _trace_lineage('gradient.brand.primary')

    assert data['gradient'] == 'gradient.brand.primary'
    assert data['css'] == (
        'linear-gradient(160deg, var(--brand-azure) 0%, var(--brand-cyan) 100%)'
    )

    primitives = {entry['token'] for entry in data['primitives']}
    assert primitives == {'color.brand.azure', 'color.brand.cyan'}

    compositions = {(entry['type'], entry['identifier']) for entry in data['compositions']}
    assert ('token', 'gradient.brand.primary') in compositions
    assert ('css-custom-property', '--brand-gradient') in compositions
    assert ('svg-gradient', '#lantern-gradient') in compositions

    usage = {(entry['type'], entry['identifier']) for entry in data['usage']}
    assert ('css-url', '#lantern-gradient') in usage

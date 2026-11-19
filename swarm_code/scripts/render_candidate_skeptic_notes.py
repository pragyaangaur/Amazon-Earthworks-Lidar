#!/usr/bin/env python3
"""Render Markdown skeptic notes from a native-reviewed candidate pack.

Adds skeptic_notes.md without changing certainty or the original automatic JSON.
Usage: .venv/bin/python code/render_candidate_skeptic_notes.py CANDIDATE_DIR
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('candidate_dir', type=Path)
    args = parser.parse_args()
    directory = args.candidate_dir
    candidate = json.loads((directory / 'candidate.json').read_text())
    if candidate.get('certainty') not in {'Strong', 'Possible'}:
        raise SystemExit('notes renderer is scoped to Strong/Possible evidence packs')
    if candidate.get('native_visual_review') is not True:
        raise SystemExit('native_visual_review is not explicitly true; refusing to synthesize notes')
    review_path = directory / 'coordinator_review.md'
    if not review_path.is_file():
        raise SystemExit('coordinator_review.md is required')
    review = review_path.read_text().strip()
    if review.startswith('# '):
        review = review.split('\n', 1)[1].lstrip() if '\n' in review else ''
    automatic_path = directory / 'skeptic_notes.json'
    automatic = json.loads(automatic_path.read_text()) if automatic_path.is_file() else {}
    automatic_state = automatic.get('status', 'not recorded')
    automatic_decision = automatic.get('decision', 'not recorded')
    literature = candidate.get('literature_checked', '')
    fid = candidate['id']
    text = f"""# Skeptic notes: {fid}

- **Certainty:** {candidate['certainty']}
- **Native visual review:** completed; see `coordinator_review.json` and `coordinator_review.md`.
- **Review timestamp:** {candidate.get('coordinator_review_utc', 'not recorded')}

## Coordinator native-resolution evaluation

{review}

## Automatic diagnostic record preserved

The original `skeptic_notes.json` is retained unchanged. Its automatic status is `{automatic_state}` and its automatic decision is `{automatic_decision}`. Those pending fields belong to the pre-review diagnostic record; they are not represented as the final native verdict. The native decision and its rationale are documented above and in `candidate.json`.

## Literature and register checks

{literature}
"""
    destination = directory / 'skeptic_notes.md'
    temporary = directory / 'skeptic_notes.md.part'
    temporary.write_text(text)
    os.replace(temporary, destination)
    print(f'WROTE {destination} ({destination.stat().st_size} bytes); automatic JSON and certainty unchanged')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

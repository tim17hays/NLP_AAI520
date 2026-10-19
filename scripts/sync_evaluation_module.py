"""Synchronize Ashley's authored function bodies without model/Drive execution."""
import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
nb = json.loads((ROOT / 'notebooks/evaluation_revision_memory.ipynb').read_text())
config_source = ''.join(nb['cells'][3]['source'])
helpers = '\n\n'.join(
    ast.get_source_segment(config_source, node)
    for node in ast.parse(config_source).body if isinstance(node, ast.FunctionDef)
)
header = '''"""Ashley Marx's current evaluator, revision loop and persistent lesson memory.

Synchronized from the authored notebook, including citation/benchmark checks
and rule-derived process lessons from verified failures. Function bodies and
prompts are preserved. Importing this module does not mount Drive/load models.
"""
import copy
import json
import os
import subprocess
import sys
import tempfile
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

CRITERIA = ("evidence_support", "completeness", "consistency", "uncertainty", "clarity")
PASSING_SCORE = 8
MAX_REVISIONS = 2
MEMORY_PATH = Path("data/processed/research_lessons.json")

'''
source = header + helpers + '\n\n' + '\n\n'.join(
    ''.join(nb['cells'][index]['source']) for index in (5, 7, 9, 11, 13, 15)
) + '\n'
source += """
# Integration transport hardening; authored functions/prompts above are unchanged.
from .model_transport import with_schema_retry

qwen_call_model = with_schema_retry(qwen_call_model, MODEL_CALL_LOG, CRITERIA)
"""
ast.parse(source)
(ROOT / 'src/evaluation_memory.py').write_text(source)
print('Synchronized authored evaluator functions; no Drive/model code executed.')

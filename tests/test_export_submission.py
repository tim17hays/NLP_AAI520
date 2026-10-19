"""Regression coverage for Colab's list-valued notebook JSON fields."""
import copy

import nbformat

from scripts.export_submission import render_notebook


def raw_colab_notebook():
    return {
        'nbformat': 4, 'nbformat_minor': 5, 'metadata': {},
        'cells': [
            {'id': 'intro', 'cell_type': 'markdown', 'metadata': {},
             'source': ['# Export test\n', 'Actual recorded output follows.']},
            {'id': 'result', 'cell_type': 'code', 'metadata': {}, 'execution_count': 1,
             'source': ['print("recorded")\n', 'print("result")'],
             'outputs': [
                 {'output_type': 'stream', 'name': 'stdout', 'text': ['recorded\n', 'result\n']},
                 {'output_type': 'display_data', 'metadata': {},
                  'data': {'text/plain': ['Score: 7\n', 'Accepted: False'],
                           'text/html': ['<table><tr><td>Score: 7</td></tr>', '</table>']}},
             ]},
        ],
    }


def test_colab_line_lists_render_without_mutating_or_dropping_outputs():
    raw = raw_colab_notebook()
    before = copy.deepcopy(raw)
    # This is the shape produced by the failed from_dict path in Colab.
    node = nbformat.from_dict(raw)
    assert isinstance(node.cells[1].source, list)
    html = render_notebook(node)
    assert raw == before
    assert 'Export test' in html
    assert 'recorded' in html and 'result' in html and 'Score: 7' in html
    assert '<table>' in html


def test_normal_string_sources_still_render():
    notebook = nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell('print("normal")')])
    html = render_notebook(notebook)
    assert 'normal' in html


def test_export_normalization_retains_actual_error_records():
    raw = raw_colab_notebook()
    error = {'output_type': 'error', 'ename': 'ValueError', 'evalue': 'actual failure',
             'traceback': ['ValueError: actual failure']}
    raw['cells'][1]['outputs'].append(error)
    before = copy.deepcopy(raw)
    html = render_notebook(raw)
    assert 'actual failure' in html
    assert raw == before

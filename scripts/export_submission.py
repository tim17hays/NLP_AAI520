"""Render the saved executed team notebook while retaining actual failures."""
import argparse
import json
from pathlib import Path
import nbformat
from nbconvert import HTMLExporter


def render_notebook(notebook):
    # Browser/Colab JSON may use line lists for cell source and output text.
    # The canonical nbformat reader rejoins these fields without mutating the
    # supplied snapshot or dropping model outputs/error records.
    notebook = nbformat.reads(json.dumps(notebook), as_version=4)
    nbformat.validate(notebook)
    body, _ = HTMLExporter(template_name='lab').from_notebook_node(notebook)
    style = '''<style>
    .jp-Notebook { max-width: 1150px; margin: auto; }
    pre, .jp-RenderedText pre, .jp-InputArea-editor pre {
      white-space: pre-wrap !important; overflow-wrap: anywhere !important;
      word-break: normal !important;
    }
    .jp-RenderedHTMLCommon table { max-width: 100%; }
    @media print {
      @page { size: A4; margin: 15mm; }
      .jp-Notebook { width: 100%; max-width: 100%; padding: 0; }
      .jp-InputArea, .jp-OutputArea { overflow: visible !important; }
      pre { font-size: 9px; }
    }
    </style>'''
    rendered = body.replace('</head>', style + '</head>')
    return '\n'.join(line.rstrip() for line in rendered.splitlines()) + '\n'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('notebook', nargs='?', default='notebooks/final_multi_agent_financial_analysis.ipynb')
    args = parser.parse_args()
    path = Path(args.notebook)
    notebook = nbformat.read(path, as_version=4)
    nbformat.validate(notebook)
    errors = [o for c in notebook.cells if c.cell_type == 'code'
              for o in c.get('outputs', []) if o.output_type == 'error']
    if errors:
        raise SystemExit(f'Notebook has {len(errors)} execution errors; inspect before exporting.')
    unexecuted = [c for c in notebook.cells if c.cell_type == 'code'
                  and c.source.strip() and c.execution_count is None]
    if unexecuted:
        raise SystemExit(f'{len(unexecuted)} code cells have not executed. Run all and save first.')
    output = path.with_suffix('.html')
    output.write_text(render_notebook(notebook), encoding='utf-8')
    print(output.resolve())


if __name__ == '__main__':
    main()

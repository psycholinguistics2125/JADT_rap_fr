#!/usr/bin/env python3
"""
PDF compilation from LaTeX and Markdown sources.
"""

from pathlib import Path
from typing import Dict, Optional

from .latex_report import generate_latex_report


def compile_latex(tex_path, output_dir=None) -> bool:
    """
    Compile a LaTeX file to PDF using pdflatex or xelatex.

    Parameters
    ----------
    tex_path : str or Path
        Path to the .tex file.
    output_dir : str or Path, optional
        Output directory for the PDF. If None, uses the same directory as tex_path.

    Returns
    -------
    bool
        True if compilation succeeded, False otherwise.
    """
    import subprocess

    tex_path = Path(tex_path).resolve()  # Use absolute path
    if output_dir is None:
        output_dir = tex_path.parent
    else:
        output_dir = Path(output_dir).resolve()  # Use absolute path

    # Try xelatex first (better Unicode support), then pdflatex
    for engine in ['xelatex', 'pdflatex']:
        try:
            # Run twice for TOC
            for _ in range(2):
                result = subprocess.run(
                    [engine, '-interaction=nonstopmode', '-output-directory', str(output_dir), str(tex_path)],
                    capture_output=True,
                    text=True,
                    cwd=str(output_dir),  # Run from output directory
                    timeout=120
                )

            if result.returncode == 0:
                print(f"LaTeX compilation successful with {engine}")
                return True
            else:
                print(f"{engine} compilation warning (returncode {result.returncode})")
                # Print last 30 lines of stderr/stdout for debugging
                if result.stderr:
                    error_lines = result.stderr.strip().split('\n')[-30:]
                    print(f"Last error lines:\n" + '\n'.join(error_lines))
                if result.stdout:
                    # Look for error lines in stdout (LaTeX outputs errors there)
                    stdout_lines = result.stdout.strip().split('\n')
                    error_lines = [l for l in stdout_lines if '!' in l or 'Error' in l or 'error' in l][-10:]
                    if error_lines:
                        print(f"LaTeX errors:\n" + '\n'.join(error_lines))
                # Check if PDF was still created
                pdf_name = tex_path.stem + '.pdf'
                pdf_path = output_dir / pdf_name
                if pdf_path.exists():
                    print(f"PDF was created despite warnings: {pdf_path}")
                    return True

        except FileNotFoundError:
            print(f"{engine} not found, trying next engine...")
            continue
        except subprocess.TimeoutExpired:
            print(f"{engine} compilation timed out")
            continue
        except Exception as e:
            print(f"{engine} compilation error: {e}")
            continue

    print("LaTeX compilation failed with all engines")
    return False


def generate_pdf_report(markdown_content: str, output_path, lang: str = 'fr',
                        pdf_engine: str = 'latex', results: Optional[Dict] = None,
                        output_dir=None, figures_dir=None) -> bool:
    """
    Generate PDF report using either LaTeX or Markdown engine.

    Parameters
    ----------
    markdown_content : str
        Markdown content (used when pdf_engine='markdown').
    output_path : str or Path
        Path for output PDF file.
    lang : str, default='fr'
        Language code for babel package.
    pdf_engine : str, default='latex'
        Engine to use: 'latex' for pure LaTeX (better equations), 'markdown' for pypandoc.
    results : dict, optional
        Results dictionary (required when pdf_engine='latex').
    output_dir : str or Path, optional
        Output directory (required when pdf_engine='latex').
    figures_dir : str or Path, optional
        Figures directory (used when pdf_engine='latex').

    Returns
    -------
    bool
        True if successful, False otherwise.
    """
    output_path = Path(output_path)

    if pdf_engine == 'latex':
        # Pure LaTeX generation
        if results is None or output_dir is None:
            print("Error: results and output_dir are required for latex engine")
            print("Falling back to markdown engine...")
            pdf_engine = 'markdown'
        else:
            try:
                # Generate LaTeX content
                tex_content = generate_latex_report(results, output_dir, figures_dir, lang)

                # Write .tex file
                tex_path = output_path.with_suffix('.tex')
                with open(tex_path, 'w', encoding='utf-8') as f:
                    f.write(tex_content)
                print(f"LaTeX source saved to: {tex_path}")

                # Compile to PDF
                if compile_latex(str(tex_path), str(output_path.parent)):
                    return True
                else:
                    print("LaTeX compilation failed, falling back to markdown engine...")
                    pdf_engine = 'markdown'
            except Exception as e:
                print(f"LaTeX generation error: {e}")
                print("Falling back to markdown engine...")
                pdf_engine = 'markdown'

    if pdf_engine == 'markdown':
        # Markdown via pypandoc
        try:
            import pypandoc
        except ImportError:
            print("Warning: pypandoc not installed. Install with: pip install pypandoc")
            return False

        babel_lang = 'french' if lang == 'fr' else 'english'

        # Resolve resource path so pandoc can find figures/
        resource_path = '.'
        if output_dir:
            resource_path = str(Path(output_dir).resolve())
        elif output_path:
            resource_path = str(Path(output_path).resolve().parent)

        extra_args = [
            '--pdf-engine=xelatex',
            '--toc',
            '--toc-depth=3',
            f'--resource-path={resource_path}',
            '-V', f'lang={babel_lang}',
            '-V', 'geometry:margin=2.5cm',
            '-V', 'fontsize=11pt',
            '-V', 'documentclass=article',
            '-V', 'mainfont=DejaVu Serif',
            '-V', 'sansfont=DejaVu Sans',
            '-V', 'monofont=DejaVu Sans Mono',
            '--highlight-style=tango',
        ]

        # Extract title from the first "# Title" line to use as YAML metadata
        # This ensures pandoc places the title BEFORE the TOC
        lines = markdown_content.split('\n')
        title_text = ''
        content_start = 0
        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped.startswith('# ') and not stripped.startswith('## '):
                title_text = stripped[2:].strip()
                content_start = i + 1
                break

        if title_text:
            # Build YAML metadata block + remaining content
            yaml_header = f"---\ntitle: \"{title_text}\"\n---\n\n"
            remaining = '\n'.join(lines[content_start:])
            pandoc_content = yaml_header + remaining
        else:
            pandoc_content = markdown_content

        try:
            pypandoc.convert_text(
                pandoc_content,
                'pdf',
                format='markdown',
                outputfile=str(output_path),
                extra_args=extra_args
            )
            return True
        except Exception as e:
            print(f"PDF generation failed: {e}")
            # Try with pdflatex as fallback
            try:
                extra_args[0] = '--pdf-engine=pdflatex'
                extra_args = [arg for arg in extra_args if 'font=' not in arg]
                pypandoc.convert_text(
                    pandoc_content,
                    'pdf',
                    format='markdown',
                    outputfile=str(output_path),
                    extra_args=extra_args
                )
                return True
            except Exception as e2:
                print(f"PDF generation with pdflatex also failed: {e2}")
                return False

    return False

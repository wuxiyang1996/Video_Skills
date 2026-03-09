from __future__ import annotations

import argparse
from pathlib import Path

import markdown as md


DEFAULT_CSS = """
@page { size: A4; margin: 18mm 16mm; }
html { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Arial, sans-serif; font-size: 11pt; }
body { line-height: 1.35; }
h1, h2, h3 { page-break-after: avoid; }
pre, code { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, "Liberation Mono", monospace; font-size: 9.5pt; }
pre {
  background: #f6f8fa;
  border: 1px solid #d0d7de;
  border-radius: 6px;
  padding: 10px 12px;
  white-space: pre-wrap;
  word-wrap: break-word;
}
code { background: #f6f8fa; padding: 0.1em 0.25em; border-radius: 4px; }
table { border-collapse: collapse; width: 100%; }
th, td { border: 1px solid #d0d7de; padding: 6px 8px; vertical-align: top; }
th { background: #f6f8fa; }
a { color: #0969da; text-decoration: none; }
hr { border: 0; border-top: 1px solid #d0d7de; }
blockquote { border-left: 3px solid #d0d7de; padding-left: 10px; color: #57606a; margin-left: 0; }
"""


def render_markdown_to_html(markdown_text: str) -> str:
    # Keep extensions lightweight but cover README conventions.
    return md.markdown(
        markdown_text,
        extensions=[
            "tables",
            "fenced_code",
            "toc",
            "sane_lists",
            "smarty",
        ],
        output_format="html5",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a review PDF from readme.md")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "readme.md",
        help="Path to Markdown input (default: repo readme.md)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "README.pdf",
        help="Output PDF path (default: README.pdf at repo root)",
    )
    parser.add_argument(
        "--also-write-html",
        action="store_true",
        help="Also write an intermediate README.html next to the PDF",
    )
    args = parser.parse_args()

    input_path: Path = args.input.resolve()
    output_pdf: Path = args.output.resolve()
    output_pdf.parent.mkdir(parents=True, exist_ok=True)

    markdown_text = input_path.read_text(encoding="utf-8")
    body_html = render_markdown_to_html(markdown_text)

    full_html = f"""<!doctype html>
<html>
  <head>
    <meta charset="utf-8" />
    <title>{input_path.name}</title>
    <style>{DEFAULT_CSS}</style>
  </head>
  <body>
    {body_html}
  </body>
</html>
"""

    output_html = output_pdf.with_suffix(".html")
    if args.also_write_html:
        output_html.write_text(full_html, encoding="utf-8")

    pdf_written = False
    weasyprint_error: str | None = None
    try:
        from weasyprint import HTML  # type: ignore

        HTML(string=full_html, base_url=str(input_path.parent)).write_pdf(str(output_pdf))
        pdf_written = True
    except Exception as e:  # pragma: no cover
        weasyprint_error = str(e)

    if not pdf_written:
        if not output_html.exists():
            output_html.write_text(full_html, encoding="utf-8")
        try:
            from playwright.sync_api import sync_playwright  # type: ignore

            with sync_playwright() as p:
                browser = p.chromium.launch()
                page = browser.new_page()
                page.goto(output_html.resolve().as_uri(), wait_until="networkidle")
                page.pdf(path=str(output_pdf), format="A4", print_background=True)
                browser.close()
            pdf_written = True
        except Exception as e:  # pragma: no cover
            if weasyprint_error:
                print(f"WeasyPrint PDF failed: {weasyprint_error}")
            print(f"Playwright PDF failed: {e}")
            print(f"Wrote HTML for manual Print-to-PDF: {output_html}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())


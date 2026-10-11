"""
Сборка PDF-отчёта из README.md.

Два момента, которые markdown сам не умеет и которые здесь обрабатываются:

  формулы  — блоки $$...$$ и вставки $...$ рендерятся через mathtext
             matplotlib в PNG и подставляются как изображения;
  графики  — встраиваются в документ как data URI, чтобы PDF был
             самодостаточным файлом, а не ссылался на папку с картинками.
"""

from __future__ import annotations

import base64
import re
import sys
from io import BytesIO
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import markdown
from weasyprint import HTML

ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"
OUT = ROOT / "reports" / "report.pdf"

CSS = """
@page {
  size: A4;
  margin: 18mm 16mm 20mm 16mm;
  @bottom-center { content: counter(page); font-size: 8.5pt; color: #888; }
}
body { font-family: "Helvetica Neue", Helvetica, Arial, sans-serif;
       font-size: 9.6pt; line-height: 1.52; color: #1a1a1a; }
h1 { font-size: 20pt; margin: 0 0 4pt 0; color: #10243a; letter-spacing: -0.3pt; }
h2 { font-size: 13pt; margin: 20pt 0 6pt 0; color: #1f4e79;
     border-bottom: 1px solid #d8dee6; padding-bottom: 3pt;
     break-after: avoid; }
h3 { font-size: 10.8pt; margin: 13pt 0 4pt 0; color: #24435f; break-after: avoid; }
p { margin: 5pt 0; }
blockquote { margin: 10pt 0; padding: 8pt 12pt; background: #f4f7fa;
             border-left: 3px solid #1f4e79; }
blockquote p { margin: 3pt 0; }
table { border-collapse: collapse; width: 100%; margin: 8pt 0;
        font-size: 8.6pt; break-inside: avoid; }
th { background: #eef2f7; text-align: left; padding: 4pt 6pt;
     border-bottom: 1.2px solid #b9c5d2; font-weight: 600; }
td { padding: 3.5pt 6pt; border-bottom: 0.5px solid #e3e8ee; vertical-align: top; }
tr:nth-child(even) td { background: #fafbfc; }
code { font-family: "SF Mono", Menlo, Consolas, monospace; font-size: 8.4pt;
       background: #f2f4f7; padding: 1pt 3pt; border-radius: 2px; }
pre { background: #f7f9fb; border: 0.5px solid #e0e6ec; border-radius: 3px;
      padding: 7pt 9pt; overflow-x: auto; break-inside: avoid; }
pre code { background: none; padding: 0; font-size: 8pt; line-height: 1.42; }
img { max-width: 100%; display: block; margin: 8pt auto; }
img.math-block { margin: 9pt auto; max-width: 78%; }
img.math-inline { display: inline; vertical-align: -0.35em; margin: 0 1pt;
                  max-height: 1.25em; }
ul, ol { margin: 5pt 0; padding-left: 17pt; }
li { margin: 2pt 0; }
hr { border: none; border-top: 0.6px solid #dde3ea; margin: 14pt 0; }
a { color: #1f4e79; text-decoration: none; }
strong { color: #0d1f30; }
"""


def _render_math(tex: str, block: bool) -> str:
    """Отрендерить формулу в data URI через mathtext."""
    size = 15 if block else 11
    fig = plt.figure(figsize=(0.01, 0.01))
    fig.text(0, 0, f"${tex}$", fontsize=size, color="#1a1a1a")
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=260, bbox_inches="tight",
                pad_inches=0.04, transparent=True)
    plt.close(fig)
    b64 = base64.b64encode(buf.getvalue()).decode()
    cls = "math-block" if block else "math-inline"
    return f'<img class="{cls}" src="data:image/png;base64,{b64}" alt="формула">'


def substitute_math(md_text: str) -> str:
    """Заменить $$...$$ и $...$ на отрендеренные изображения."""
    protected: list[str] = []

    def stash(m):
        protected.append(m.group(0))
        return f"\x00PROT{len(protected) - 1}\x00"

    # код не трогаем: там $ может быть частью текста
    md_text = re.sub(r"```.*?```", stash, md_text, flags=re.S)
    md_text = re.sub(r"`[^`\n]+`", stash, md_text)

    md_text = re.sub(r"\$\$(.+?)\$\$",
                     lambda m: "\n\n" + _render_math(m.group(1).strip(), True) + "\n\n",
                     md_text, flags=re.S)
    md_text = re.sub(r"(?<!\$)\$([^$\n]+?)\$(?!\$)",
                     lambda m: _render_math(m.group(1).strip(), False), md_text)

    for i, blk in enumerate(protected):
        md_text = md_text.replace(f"\x00PROT{i}\x00", blk)
    return md_text


def embed_images(html: str) -> str:
    """Встроить локальные картинки как data URI — PDF должен быть самодостаточным."""
    def repl(m):
        src = m.group(1)
        if src.startswith(("http", "data:")):
            return m.group(0)
        p = (ROOT / src).resolve()
        if not p.exists():
            return ""                       # отсутствующий график не ломает сборку
        b64 = base64.b64encode(p.read_bytes()).decode()
        return m.group(0).replace(src, f"data:image/png;base64,{b64}")
    return re.sub(r'<img[^>]*src="([^"]+)"', repl, html)


def build(verbose: bool = True) -> Path:
    md_text = README.read_text(encoding="utf-8")
    md_text = substitute_math(md_text)
    html_body = markdown.markdown(
        md_text,
        extensions=["tables", "fenced_code", "codehilite", "attr_list", "sane_lists"],
        extension_configs={"codehilite": {"noclasses": True, "pygments_style": "friendly"}},
    )
    html_body = embed_images(html_body)
    html = (f'<!doctype html><html lang="ru"><head><meta charset="utf-8">'
            f"<style>{CSS}</style></head><body>{html_body}</body></html>")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    HTML(string=html, base_url=str(ROOT)).write_pdf(OUT)
    if verbose:
        print(f"  {OUT.relative_to(ROOT)}  ({OUT.stat().st_size / 1024:.0f} КиБ)")
    return OUT


if __name__ == "__main__":
    build()

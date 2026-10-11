"""
Рендер ключевых фрагментов исходного кода в картинки для отчёта.

Показываются не случайные куски, а те места, где собственно и находится
содержание работы: рекуррента моментов возрастов, порядок операций,
исключающий утечку, горячий цикл движка и тест с позитивным контролем.
Фрагменты задаются диапазоном строк и подписью, и берутся прямо из
исходников — рассинхронизироваться с репозиторием не могут.
"""

from __future__ import annotations

from pathlib import Path

from pygments import highlight
from pygments.formatters import ImageFormatter
from pygments.lexers import PythonLexer

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "screenshots"

FONT = "Menlo"

#: (файл, маркер начала, сколько строк, имя файла, подпись)
SNIPPETS = [
    ("src/afe/kernels.py", "def age_moment_decay_step(", 8,
     "code_моменты_возрастов.png",
     "рекуррента моментов возраста, O(1) на событие"),
    ("src/afe/log.py", "    q_rank = np.int8(2)", 14,
     "code_порядок_без_утечки.png",
     "запрос раньше события при равном timestamp"),
    ("src/afe/engine.py", "        else:\n            # ---- ОБНОВЛЕНИЕ", 22,
     "code_движок_обновление.png",
     "горячий цикл: обновляются только слоты своего типа события"),
    ("src/afe/schema.py", "    def key_specs(", 20,
     "code_перебор_мета_путей.png",
     "перечисление ключей = перечисление мета-путей"),
    ("tests/test_engine.py", "def test_leak_detector_actually_detects_leak(", 22,
     "code_позитивный_контроль.png",
     "позитивный контроль: тест обязан ловить заведомую утечку"),
]


def render_snippet(path: str, marker: str, n_lines: int, out_name: str) -> Path | None:
    src = (ROOT / path).read_text(encoding="utf-8")
    idx = src.find(marker)
    if idx < 0:
        print(f"  {path}: маркер не найден, пропуск")
        return None
    start_line = src[:idx].count("\n")
    lines = src.splitlines()[start_line:start_line + n_lines]
    code = "\n".join(lines)

    fmt = ImageFormatter(
        font_name=FONT, font_size=17, line_numbers=True,
        line_number_start=start_line + 1, line_number_bg="#20232b",
        line_number_fg="#6b7384", line_number_separator=False,
        line_pad=6, image_pad=18, style="monokai",
    )
    png = highlight(code, PythonLexer(), fmt)
    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / out_name
    out.write_bytes(png)
    return out


def render_all(verbose: bool = True) -> list[Path]:
    made = []
    for path, marker, n, name, caption in SNIPPETS:
        p = render_snippet(path, marker, n, name)
        if p:
            made.append(p)
            if verbose:
                print(f"  {p.relative_to(ROOT)}  — {caption}")
    return made


if __name__ == "__main__":
    render_all()

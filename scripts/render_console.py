"""
Оформление РЕАЛЬНОГО вывода экспериментов в картинки для отчёта.

Берёт лог конкретного запуска и рисует его в виде окна терминала. Ничего
не придумывает и не редактирует: показывается ровно то, что напечатала
программа. Нужно затем, что таблицы метрик и профили кривых в отчёте
читаются гораздо лучше картинкой, чем кодовым блоком на пол-страницы.

Запуск:
    python scripts/render_console.py logs/e05.log "make e05"
    python scripts/render_console.py --all
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "screenshots"

FONT_PATH = "/System/Library/Fonts/Menlo.ttc"
FONT_SIZE = 15
LINE_H = 21
PAD = 20
TITLEBAR_H = 34
#: Ширина в символах. Подобрана по самой длинной строке вывода — таблице
#: бенчмарка с доверительными интервалами (около 120 символов). При меньшем
#: значении строки переносятся пополам и таблица в отчёте выглядит
#: сломанной.
MAX_COLS = 128

# палитра в духе тёмной темы терминала
BG = (24, 26, 32)
BG_BAR = (38, 41, 50)
FG = (214, 219, 228)
DIM = (122, 130, 145)
ACCENT = FG                   # числа и метрики: без отдельного цвета
GREEN = (126, 211, 143)       # приглашение и команда
WARN = (240, 180, 110)        # предупреждения и вердикты
DOTS = ((255, 95, 86), (255, 189, 46), (39, 201, 63))

RE_NUM = re.compile(r"(?<![\w.])(\d+[.,]\d+|\d{2,})(?![\w.])")
RE_HEAD = re.compile(r"^[=\-]{6,}$")


def _font(bold: bool = False) -> ImageFont.FreeTypeFont:
    try:
        return ImageFont.truetype(FONT_PATH, FONT_SIZE, index=1 if bold else 0)
    except Exception:
        return ImageFont.load_default()


def _wrap(lines: list[str], width: int) -> list[str]:
    out: list[str] = []
    for ln in lines:
        ln = ln.rstrip("\n").replace("\t", "    ")
        while len(ln) > width:
            out.append(ln[:width])
            ln = ln[width:]
        out.append(ln)
    return out


def _line_color(text: str) -> tuple[int, int, int]:
    t = text.strip()
    if RE_HEAD.match(t):
        return DIM
    if t.startswith(("===", "---")):
        return DIM
    if "ВЕРДИКТ" in t or "ВНИМАНИЕ" in t or "УТЕЧК" in t.upper():
        return WARN
    if t.startswith("["):
        return ACCENT
    return FG


def render(log_path: Path, command: str, out_path: Path,
           head: int | None = None, tail: int | None = None,
           lines_range: tuple[int, int] | None = None,
           title: str | None = None) -> Path:
    """lines_range задаёт произвольный кусок лога (1-индексация, включительно).

    Нужен, когда важная часть вывода не в начале и не в конце: например,
    в e08 сравнение джойнов идёт первым блоком, а эффект разрыва
    предсказания — последним, и одной картинкой их не показать, не
    превысив высоту страницы A4.
    """
    raw = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    if lines_range:
        a, b = lines_range
        raw = raw[a - 1:b]
    elif head:
        raw = raw[:head]
    elif tail:
        raw = raw[-tail:]
    lines = _wrap(raw, MAX_COLS)

    font, font_b = _font(), _font(bold=True)
    body = [f"$ {command}", ""] + lines
    width = PAD * 2 + int(max(font.getlength(l) for l in body) if body else 400)
    width = max(width, 560)
    height = TITLEBAR_H + PAD * 2 + LINE_H * len(body)

    img = Image.new("RGB", (width, height), BG)
    d = ImageDraw.Draw(img)

    # заголовок окна
    d.rectangle([0, 0, width, TITLEBAR_H], fill=BG_BAR)
    for i, c in enumerate(DOTS):
        cx = 18 + i * 19
        d.ellipse([cx - 6, TITLEBAR_H // 2 - 6, cx + 6, TITLEBAR_H // 2 + 6], fill=c)
    cap = title or f"{log_path.name} — tbank-recsys-autofe"
    d.text((width // 2 - font.getlength(cap) / 2, TITLEBAR_H // 2 - FONT_SIZE // 2 - 1),
           cap, font=font, fill=DIM)

    y = TITLEBAR_H + PAD
    for idx, text in enumerate(body):
        if idx == 0:                       # строка приглашения с командой
            d.text((PAD, y), "$", font=font_b, fill=GREEN)
            d.text((PAD + font.getlength("$ "), y), text[2:], font=font_b, fill=GREEN)
            y += LINE_H
            continue
        base = _line_color(text)
        x = PAD
        pos = 0
        for m in RE_NUM.finditer(text):     # числа подсвечиваются
            if m.start() > pos:
                seg = text[pos:m.start()]
                d.text((x, y), seg, font=font, fill=base)
                x += font.getlength(seg)
            d.text((x, y), m.group(), font=font, fill=ACCENT if base is FG else base)
            x += font.getlength(m.group())
            pos = m.end()
        if pos < len(text):
            d.text((x, y), text[pos:], font=font, fill=base)
        y += LINE_H

    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)
    return out_path


#: Что рендерить при --all: (лог, команда, имя файла, строк с конца).
#:
#: Хвосты ограничены ~30 строками не для красоты: картинка высотой в 1000
#: пикселей на странице A4 не помещается и ломает вёрстку PDF. Берётся
#: итоговая часть вывода — таблица результатов и вывод, — а не весь лог;
#: полный лог лежит рядом в logs/ и ничего не теряется.
TARGETS = [
    ("logs/e08.log", "make e08", "08a_цена_тихой_утечки.png", None, (4, 15)),
    ("logs/e01.log", "make e01", "01_окна_против_затуханий.png", 22),
    ("logs/e02.log", "make e02", "02_временные_масштабы.png", 30),
    ("logs/e05.log", "make e05", "05_аудит_утечки.png", 26),
    ("logs/e06.log", "make e06", "06_масштабирование.png", 24),
    ("logs/e07.log", "make e07", "07_отбор_под_бюджет.png", 26),
    ("logs/e04.log", "make e04", "04_сравнение_с_агентом.png", 24),
    ("logs/e03_retailrocket.log", "make e03-rr", "03_бенчмарк_retailrocket.png", 24),
    ("logs/e03_synthetic.log", "make e03", "03_бенчмарк_синтетика.png", 24),
    ("logs/tests.log", "make test", "00_тесты.png", 8),
    ("logs/e08.log", "make e08", "08_диагностика_протокола.png", 34),
]


def render_all(verbose: bool = True) -> list[Path]:
    made = []
    for entry in TARGETS:
        rel, cmd, name, tail = entry[:4]
        rng = entry[4] if len(entry) > 4 else None
        p = ROOT / rel
        if not p.exists():
            if verbose:
                print(f"  {rel}: нет лога, пропуск")
            continue
        out = render(p, cmd, OUT / name, tail=tail, lines_range=rng)
        made.append(out)
        if verbose:
            print(f"  {out.relative_to(ROOT)}")
    return made


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("log", nargs="?")
    ap.add_argument("command", nargs="?")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--tail", type=int, default=40)
    ap.add_argument("--out")
    a = ap.parse_args()
    if a.all or not a.log:
        render_all()
    else:
        p = Path(a.log)
        out = Path(a.out) if a.out else OUT / f"{p.stem}.png"
        print(render(p, a.command or f"python {p.stem}.py", out, tail=a.tail))

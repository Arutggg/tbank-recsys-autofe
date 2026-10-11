# Воспроизведение всей работы. Каждая цель самодостаточна.
PY := ./.venv/bin/python
export PYTHONPATH := src

.PHONY: help setup test e01 e02 e03 e03-rr e04 e05 e06 e07 figures fill report all data clean

help:
	@echo "setup    — окружение и зависимости"
	@echo "test     — тесты корректности (включая аудит утечки)"
	@echo "data     — скачать RetailRocket (нужен токен Kaggle)"
	@echo "e01..e07 — отдельные эксперименты"
	@echo "figures  — графики по сохранённым результатам"
	@echo "all      — тесты, все эксперименты, графики"

setup:
	python3.12 -m venv .venv
	$(PY) -m pip install -q --upgrade pip
	$(PY) -m pip install -q -r requirements.txt
	@echo "LightGBM на macOS требует libomp: brew install libomp"

test:
	$(PY) -m pytest tests/ -q

data:
	@test -f ~/.kaggle/kaggle.json || \
	  (echo "нет ~/.kaggle/kaggle.json — создайте токен на kaggle.com/settings"; exit 1)
	mkdir -p data/retailrocket
	./.venv/bin/kaggle datasets download -d retailrocket/ecommerce-dataset \
	  -p data/retailrocket --unzip

e01:
	$(PY) experiments/e01_window_vs_decay.py
e02:
	$(PY) experiments/e02_timescales.py
e03:
	$(PY) experiments/e03_benchmark.py --dataset synthetic
e03-rr:
	$(PY) experiments/e03_benchmark.py --dataset retailrocket \
	  --n-positives 100000 --n-positives-train 100000 \
	  --n-negatives-eval 49 --n-trials 8
e04:
	$(PY) experiments/e04_vs_agent.py
e05:
	$(PY) experiments/e05_leak_audit.py
e06:
	$(PY) experiments/e06_scaling.py
e07:
	$(PY) experiments/e07_budget.py

figures:
	$(PY) -m afe.plots

fill:
	$(PY) scripts/fill_results.py

report: figures fill
	$(PY) scripts/build_pdf.py

all: test e01 e02 e05 e06 e07 e04 e03 e03-rr figures report

clean:
	rm -rf reports/*.json reports/figures/*.png .pytest_cache
	find . -name __pycache__ -type d -exec rm -rf {} +

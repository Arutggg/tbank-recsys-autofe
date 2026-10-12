# Воспроизведение всей работы. Каждая цель самодостаточна.
PY := ./.venv/bin/python
export PYTHONPATH := src


.PHONY: all clean data e01 e02 e03 e03-rr e04 e05 e06 e07 e08 figures fill help lint report screenshots setup test

help:
	@echo "setup    — окружение и зависимости"
	@echo "lint     — ruff"
	@echo "test     — линтер + тесты корректности (включая аудит утечки)"
	@echo "data     — скачать RetailRocket (нужен токен Kaggle)"
	@echo "e01..e08 — отдельные эксперименты"
	@echo "figures  — графики по сохранённым результатам"
	@echo "screenshots — вывод экспериментов в виде картинок терминала"
	@echo "all      — тесты, все эксперименты, графики"

setup:
	python3.12 -m venv .venv
	$(PY) -m pip install -q --upgrade pip
	$(PY) -m pip install -q -r requirements.txt
	@echo "LightGBM на macOS требует libomp: brew install libomp"

lint:
	$(PY) -m ruff check .

test: lint
	@mkdir -p logs
	$(PY) -m pytest tests/ -q 2>&1 | tee logs/tests.log

data:
	@test -f ~/.kaggle/kaggle.json || \
	  (echo "нет ~/.kaggle/kaggle.json — создайте токен на kaggle.com/settings"; exit 1)
	mkdir -p data/retailrocket
	./.venv/bin/kaggle datasets download -d retailrocket/ecommerce-dataset \
	  -p data/retailrocket --unzip

e01:
	@mkdir -p logs
	$(PY) experiments/e01_window_vs_decay.py 2>&1 | tee logs/e01.log
e02:
	@mkdir -p logs
	$(PY) experiments/e02_timescales.py 2>&1 | tee logs/e02.log
e03:
	@mkdir -p logs
	$(PY) experiments/e03_benchmark.py --dataset synthetic \
	  --n-negatives-eval 49 --prediction-gap-days 1 --n-trials 8 2>&1 | tee logs/e03_synthetic.log
e03-rr:
	@mkdir -p logs
	$(PY) experiments/e03_benchmark.py --dataset retailrocket \
	  --n-positives 100000 --n-positives-train 100000 \
	  --n-negatives-eval 49 --prediction-gap-days 1 --n-trials 8 \
	  2>&1 | tee logs/e03_retailrocket.log
e04:
	@mkdir -p logs
	$(PY) experiments/e04_vs_agent.py 2>&1 | tee logs/e04.log
e05:
	@mkdir -p logs
	$(PY) experiments/e05_leak_audit.py 2>&1 | tee logs/e05.log
e06:
	@mkdir -p logs
	$(PY) experiments/e06_scaling.py 2>&1 | tee logs/e06.log
e07:
	@mkdir -p logs
	$(PY) experiments/e07_budget.py 2>&1 | tee logs/e07.log
e08:
	@mkdir -p logs
	$(PY) experiments/e08_protocol_diagnostics.py 2>&1 | tee logs/e08.log

figures:
	$(PY) -m afe.plots

screenshots:
	$(PY) scripts/render_console.py --all
	$(PY) scripts/render_code.py

fill:
	$(PY) scripts/fill_results.py

report: figures screenshots fill
	$(PY) scripts/build_pdf.py

all: test e01 e02 e08 e05 e06 e07 e04 e03 e03-rr figures report

clean:
	rm -rf reports/*.json reports/figures/*.png .pytest_cache
	find . -name __pycache__ -type d -exec rm -rf {} +

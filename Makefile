.PHONY: setup run-app test fmt lint precommit embed-index

setup:
python3 -m venv .venv && . .venv/bin/activate && pip install -U pip && pip install -r requirements.txt && pre-commit install

run-app:
. .venv/bin/activate && streamlit run app/Home.py

test:
. .venv/bin/activate && pytest -q

fmt:
. .venv/bin/activate && ruff format .

lint:
. .venv/bin/activate && ruff check .

precommit:
. .venv/bin/activate && pre-commit run --all-files

embed-index:
. .venv/bin/activate && python -m cbrlab.hybrid.embeddings build

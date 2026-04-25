.PHONY: install run test evaluate lint clean

install:
	pip install -r requirements.txt

run:
	streamlit run ui/streamlit_app.py

test:
	pytest tests/ -v

evaluate:
	python -m evaluation.rag_evaluator

lint:
	python -m py_compile config.py
	python -m py_compile embeddings/embedder.py
	python -m py_compile retrieval/rag_pipeline.py
	python -m py_compile sentiment/sentiment_analyzer.py

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache

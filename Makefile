setup:
	python scripts/download_data.py
	python scripts/prepare_data.py
	python scripts/train_tokenizers.py
train:
	python scripts/train.py
evaluate:
	python scripts/evaluate.py
ui:
	streamlit run app/app.py
test:
	python scripts/run_tests.py

.PHONY: install data train fast app deploy-deps clean help

help:
	@echo "make install   - install full dependencies"
	@echo "make data      - download the NHTSA complaints corpus"
	@echo "make train     - full pipeline: build splits, train 3 models, run experiments"
	@echo "make fast      - tiny smoke-test run (seconds) to verify the plumbing"
	@echo "make app       - launch the Flask web app at http://localhost:5000"
	@echo "make clean     - remove generated data/outputs and trained models"

install:
	pip install -r requirements.txt

data:
	python scripts/make_dataset.py

train:
	python setup.py

fast:
	python setup.py --fast

app:
	python main.py

deploy-deps:
	pip install -r requirements-deploy.txt

clean:
	rm -rf data/processed/*.csv data/outputs/*.json data/outputs/plots/*.png models/*.pkl models/*.pt models/labels.json

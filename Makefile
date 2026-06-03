dirs:
	mkdir -p data/raw
	mkdir -p data/preprocess
	mkdir -p data/models

download: dirs
	wget -O data/raw/b3db.tsv https://raw.githubusercontent.com/theochem/B3DB/refs/heads/main/B3DB/B3DB_classification.tsv

features:
	python features.py 
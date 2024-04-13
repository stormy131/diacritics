all:
	mkdir data data/eval data/train
	curl https://ufal.mff.cuni.cz/~zabokrtsky/courses/npfl124/data/diacritics-dtest.txt > data/eval/diacritics-dtest.txt
	curl https://ufal.mff.cuni.cz/~zabokrtsky/courses/npfl124/data/diacritics-etest.txt > data/eval/diacritics-etest.txt
	python -m venv venv
	. ./venv/bin/activate
	pip install -r requirements.txt

	python training.py

clean:
	deactivate
	rm -rf __pycache__
	rm -rf venv
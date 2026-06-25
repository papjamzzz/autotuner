run:
	python3 app.py

install:
	pip3 install -r requirements.txt

push:
	git add -A && git commit -m "$(m)" && git push

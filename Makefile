.PHONY: format
format:
	poetry run black src tests noxfile.py

.PHONY: lint
lint:
	poetry run flake8 src tests noxfile.py

.PHONY: test
test:
	poetry run pytest --doctest-modules src tests

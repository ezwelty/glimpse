.PHONY: format
format:
	poetry run black src tests

.PHONY: lint
lint:
	poetry run flake8 src tests

.PHONY: test
test:
	poetry run pytest --doctest-modules src tests

.PHONY: format
format:
	poetry run black src tests

.PHONY: lint
lint:
	poetry run flake8 src tests

.PHONY: test
test:
	poetry run pytest --doctest-modules src tests

.PHONY: testcov
testcov:
	poetry run pytest --cov --doctest-modules src tests

.PHONY: pushcov
pushcov:
	poetry run coverage xml --fail-under=0
	poetry run codecov

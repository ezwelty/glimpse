.PHONY: format lint test testcov pushcov docs

format:
	poetry run isort src tests docs
	poetry run black src tests docs

lint:
	poetry run flake8 src tests docs

test:
	poetry run pytest --doctest-modules src tests

testcov:
	poetry run pytest --cov --doctest-modules src tests

pushcov:
	poetry run coverage xml --fail-under=0
	poetry run codecov

docs:
	poetry run sphinx-build docs docs/build

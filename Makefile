.PHONY: quality style test test-examples

# Check that source code meets quality standards

quality:
	black --check --line-length 119 --target-version py38 .
	isort --check-only .
	flake8 --max-line-length 119

# Format source code automatically

style:
	black --line-length 119 --target-version py38 .
	isort .

test:
	pytest -sv ./src/
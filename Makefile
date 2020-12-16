.PHONY: quality style test test-examples

# Check that source code meets quality standards

quality:
	black --check --line-length 119 --target-version py36 tests src
	isort --check-only tests src
	flake8 tests src

# Format source code automatically

style:
	black --line-length 119 --target-version py36 tests src
	isort tests src
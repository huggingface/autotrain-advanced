.PHONY: quality style test

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

docker:
	docker build -t autotrain-advanced:latest .
	docker tag autotrain-advanced:latest huggingface/autotrain-advanced:latest
	docker push huggingface/autotrain-advanced:latest

api:
	docker build -t autotrain-advanced-api:latest -f Dockerfile.api .
	docker tag autotrain-advanced-api:latest public.ecr.aws/z4c3o6n6/autotrain-api:latest
	docker push public.ecr.aws/z4c3o6n6/autotrain-api:latest

ngc:
	docker build -t autotrain-advanced:latest .
	docker tag autotrain-advanced:latest nvcr.io/ycymhzotssoi/autotrain-advanced:latest
	docker push nvcr.io/ycymhzotssoi/autotrain-advanced:latest

pip:
	rm -rf build/
	rm -rf dist/
	make style && make quality
	python setup.py sdist bdist_wheel
	twine upload dist/* --verbose
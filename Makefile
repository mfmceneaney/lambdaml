install:
	pip install --upgrade pip &&\
		pip install -e .[test]

format:
	black core/lambdaml/*.py core/tests/*.py pyscripts/*.py app/*.py

lint:
	pylint --max-args=1000 \
		--max-positional-arguments=1000 \
		--max-locals=100 \
		--max-line-length=200 \
		--max-module-lines=2000 \
		--max-statements=100 \
		--max-branches=20 \
		--disable=R,C core/lambdaml/*.py core/tests/*.py pyscripts/*.py app/*.py

test:
	python -m pytest core/tests

all: install lint test

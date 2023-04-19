PKGNAME=km3buu
ALLNAMES = $(PKGNAME)

default: run

all: install

install:
	pip install setuptools_scm
	pip install .

install-dev:
	pip install setuptools_scm
	pip install -e ".[dev]"
	pip install -e ".[extras]"	

test:
	python -m pytest --junitxml=./reports/junit.xml -o junit_suite_name=$(PKGNAME) $(PKGNAME)

test-cov:
	python -m pytest --cov ./ --cov-report term-missing --cov-report xml:reports/coverage.xml --cov-report html:reports/coverage $(ALLNAMES)

flake8: 
	python -m pytest --flake8

docstyle: 
	python -m pytest --pydocstyle

doc:
	cd doc && make html
	cd ..

.PHONY: install install-dev doc clean test test-cov flake8 docstyle

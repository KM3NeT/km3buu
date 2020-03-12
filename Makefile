PKGNAME=km3buu
ALLNAMES = $(PKGNAME)

export REPO_OUTPUT_DIR := output
export REPO_JOBCARDS_DIR := jobcards
export CONTAINER_OUTPUT_DIR := /opt/output
export CONTAINER_JOBCARD_DIR := /opt/jobcards

default: run

build: km3buu.Singularity
	sudo singularity build GiBUU.simg km3buu.Singularity 

run: GiBUU.simg
	@if [ ! -d "jobcards/${CARDSET}" ];then \
	    exit 1; \
        fi;
	@if [ -d "${REPO_OUTPUT_DIR}/${CARDSET}" ];then \
	    echo "Output directory for this cardset already exists; remove and go on [y/n]";\
	    read REPLY; \
	    if [ ! $$REPLY = "y" ];then \
		exit 2;\
	    fi;\
	fi;
	rm -rf ${REPO_OUTPUT_DIR}/${CARDSET};
	mkdir -p ${REPO_OUTPUT_DIR}/${CARDSET};
	singularity exec -B ${REPO_JOBCARDS_DIR}/${CARDSET}:$$CONTAINER_JOBCARD_DIR\
			 -B ${REPO_OUTPUT_DIR}/${CARDSET}:$$CONTAINER_OUTPUT_DIR\
			 GiBUU.simg\
			 /bin/sh run.sh $$CONTAINER_JOBCARD_DIR $$CONTAINER_OUTPUT_DIR 

buildremote:
	singularity build GiBUU.simg docker://docker.km3net.de/simulation/km3buu:latest

clean:
	@rm -rf output
	python setup.py clean --all

### PYTHON ###
install:
	pip install .

install-dev:
	pip install -e ".[dev]"

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

.PHONY: install install-dev doc clean test test-cov flake8 docstyle buildremote

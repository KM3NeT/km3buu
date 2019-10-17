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

lyon:
	@echo "TO DO"

clean:
	@rm -rf output



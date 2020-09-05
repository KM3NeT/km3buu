#!/bin/bash

if [ -z "$CONTAINER_GIBUU_EXEC+x" ];
then 
    echo "No GIBUU executable provided via CONTAINER_GIBUU_EXEC";
    exit 1
fi;

if [ $# -eq 0 ];
then
    echo "No paths provided!";
    exit 1;
elif [ $# -eq 1 ];
then
    echo "No output directory within the container given!";
    exit 1;
fi;

CONTAINER_JOBCARD_DIR=$1
CONTAINER_OUTPUT_DIR=$2

JOBCARDS=$(find $CONTAINER_JOBCARD_DIR -name "*.job")

cd $CONTAINER_OUTPUT_DIR

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/bin

for filepath in $JOBCARDS; do
    filename=$(basename -- $filepath)
    foldername="${filename%.*}"
    mkdir $foldername; cd $foldername
    $CONTAINER_GIBUU_EXEC < $filepath;
    cd ..
done

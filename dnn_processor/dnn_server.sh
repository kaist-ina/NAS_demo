#!/bin/bash

source activate nas

while getopts ":g:q:c:d:" opt; do
    case "$opt" in
    g)
        gpu=$OPTARG
        ;;
    q)
        quality=$OPTARG
        ;;
    c)
        contentType=$OPTARG
        ;;
    d)
        dataType=$OPTARG
        ;;
    *)
        echo "Usage: $0 -g [gpu] -c [contentType] -q [quality] -d [dataType]" 1>&2; exit 1;
        ;;
    esac
done

if [ $OPTIND -eq 1 ]; then echo "Usage: $0 -g [gpu] -c [contentType] -q [quality] -d [dataType]" 1>&2; exit 1; fi
if [ $OPTIND -eq 3 ]; then echo "Usage: $0 -g [gpu] -c [contentType] -q [quality] -d [dataType]" 1>&2; exit 1; fi
if [ $OPTIND -eq 5 ]; then echo "Usage: $0 -g [gpu] -c [contentType] -q [quality] -d [dataType]" 1>&2; exit 1; fi
if [ $OPTIND -eq 7 ]; then echo "Usage: $0 -g [gpu] -c [contentType] -q [quality] -d [dataType]" 1>&2; exit 1; fi


rm -rf downloads
#./dnn_server.sh -g 0 -c news -q ultra -d video
CUDA_VISIBLE_DEVICES=$gpu python dnn_server.py --contentType $contentType --quality $quality --testEpoch 10 --dataType $dataType
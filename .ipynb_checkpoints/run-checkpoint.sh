#!/bin/bash

while getopts ":i:o:" opt; do
  case $opt in
    i) inputDataset="$OPTARG"
    ;;
    o) outputDir="$OPTARG"
    ;;
    \?) echo "Invalid option -$OPTARG" >&2
    exit 1
    ;;
  esac

  case $OPTARG in
    -*) echo "Option $opt needs a valid argument"
    exit 1
    ;;
  esac
done

python3 query_parsing_classification.py $inputDataset/topics.xml
python3 query_processing.py $inputDataset/topics.xml
screen elasticsearch
gzip -d  $inputDataset/passages.jsonl.gz
python3 elasticsearch_initial_document_retrieval.py $inputDataset/topics.xml $inputDataset/passages.jsonl
python3 retrieve_and_extract.py ../$inputDataset/topics.xml
python3 argument_quality_improv.py ../$inputDataset/topics.xml
python3 index_and_rank.py ../$inputDataset/topics.xml
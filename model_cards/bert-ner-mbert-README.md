---
language:
- multilingual
- en
license: mit
library_name: transformers
tags:
- bert
- token-classification
- ner
- encoder-only
datasets:
- wikiann
---

# sajida12/bert-ner-mbert

Encoder-only multilingual BERT fine-tuned for **Named Entity Recognition (NER)**.

## Task
Find entities such as **Person (PER)**, **Organization (ORG)**, and **Location (LOC)**.  
This is text understanding — **not sentiment analysis**.

## Base model
`bert-base-multilingual-cased` (mBERT)

## Dataset
WikiANN (high-resource / multilingual NER)

## Intended use
- Short sentence input
- Token / entity prediction in a Streamlit app section

## Project links
- GitHub: https://github.com/sajida12-sys/curecauprojiect
- Space card: https://huggingface.co/spaces/sajida12/balochi-gpt-test

---
language:
- bal
license: mit
library_name: transformers
tags:
- gpt2
- text-generation
- balochi
- decoder-only
- low-resource
---

# sajida12/gpt2-balochi

Decoder-only GPT-2 adapted for **Balochi-only** (low-resource, monolingual) text generation / Q&A.

## Task
Balochi question → Balochi answer / continuation.  
Language is **Balochi only** (not Urdu or other languages).

## Base model
`gpt2`

## Dataset
Balochi instruction-response pairs from the project dataset (`balochi_instructions.csv`).

## Intended use
- Short Balochi prompt input
- Generation / Q&A in the Streamlit app GPT section

## Project links
- GitHub: https://github.com/sajida12-sys/curecauprojiect
- Space card: https://huggingface.co/spaces/sajida12/balochi-gpt-test

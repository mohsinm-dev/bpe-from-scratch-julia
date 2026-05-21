# BPE from Scratch — Julia

A minimal implementation of Byte Pair Encoding (BPE) in Julia, built step by step for learning purposes.

## What is BPE?

Byte Pair Encoding is a subword tokenization algorithm used in modern language models (GPT, etc.). It iteratively merges the most frequent pair of adjacent symbols to build a vocabulary.

## Usage

```julia
include("src/BytePairEncoding.jl")
using .BytePairEncoding

corpus = "low low low lower lower lowest"
vocab, merges = train_bpe(corpus, 10)

for (i, merge) in enumerate(merges)
    println("$i. $(merge[1]) + $(merge[2])")
end
```

## Functions

- `word_to_symbols(word)` — split a word into characters with an end-of-word marker
- `count_word_frequencies(corpus)` — count word occurrences in a corpus
- `initialize_word_symbols(freqs)` — convert words to initial symbol sequences
- `count_pairs(word_symbols)` — count adjacent symbol pairs weighted by frequency
- `best_pair(pair_counts)` — find the most frequent pair
- `merge_symbols(symbols, pair)` — merge all occurrences of a pair in a symbol sequence
- `train_bpe(corpus, num_merges)` — run the full BPE training loop

## Running tests

```bash
julia test/runtests.jl
```

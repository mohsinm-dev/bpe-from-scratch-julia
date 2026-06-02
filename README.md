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

### Encoding new text

```julia
tokens = encode_text("low lower", merges)
println(tokens)           # ["low</w>", "lower</w>"]
println(decode_tokens(tokens))  # "low lower"
```

### Save and load trained merges

```julia
save_merges(merges, "merges.tsv")
loaded_merges = load_merges("merges.tsv")
```

### Training options

```julia
# Verbose mode shows each merge step
vocab, merges = train_bpe(corpus, 10, verbose=true)

# Stop early when pair frequency drops below threshold
vocab, merges = train_bpe(corpus, 100, min_frequency=3)
```

### Analytics

```julia
tokens = encode_text("low lower lowest", merges)
ratio = compression_ratio("low lower lowest", tokens)
freqs = token_frequencies(tokens)
history = vocab_size_history(corpus, 10)
```

### Batch encoding and BPE dropout

```julia
results = encode_batch(["low", "lower", "lowest"], merges)

# Stochastic tokenization for training robustness
tokens = encode_word_with_dropout("lower", merges, dropout=0.1)
```

## Functions

### Core training
- `word_to_symbols(word)` — split a word into characters with an end-of-word marker
- `count_word_frequencies(corpus)` — count word occurrences in a corpus
- `initialize_word_symbols(freqs)` — convert words to initial symbol sequences
- `count_pairs(word_symbols)` — count adjacent symbol pairs weighted by frequency
- `best_pair(pair_counts)` — find the most frequent pair
- `merge_symbols(symbols, pair)` — merge all occurrences of a pair in a symbol sequence
- `train_bpe(corpus, num_merges; verbose, min_frequency)` — run the full BPE training loop
- `get_vocabulary(word_symbols)` — extract unique tokens from trained vocabulary

### Encoding and decoding
- `encode_word(word, merges)` — tokenize a single word using learned merges
- `encode_text(text, merges)` — tokenize a full text string
- `decode_tokens(tokens)` — reconstruct text from BPE tokens
- `encode_batch(texts, merges)` — encode multiple texts at once
- `encode_word_with_dropout(word, merges; dropout)` — stochastic tokenization

### Preprocessing and I/O
- `preprocess_text(text; lowercase)` — normalize text for training
- `load_corpus(filepath)` — read a corpus from file
- `save_merges(merges, filepath)` — write merge rules to file
- `load_merges(filepath)` — read merge rules from file
- `save_vocab(vocab, filepath)` — export vocabulary to file

### Analytics
- `compression_ratio(text, tokens)` — characters per token ratio
- `token_frequencies(tokens)` — count token occurrences
- `vocab_size_history(corpus, num_merges)` — track vocab size over training
- `add_special_tokens(vocab, special)` — add special tokens to vocabulary

## Running tests

```bash
julia test/runtests.jl
```

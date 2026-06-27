# BPE from Scratch — Julia

A complete implementation of subword tokenization algorithms in Julia — BPE, WordPiece, and Unigram — built from scratch for learning and experimentation.

## What is BPE?

Byte Pair Encoding is a subword tokenization algorithm used in modern language models (GPT, etc.). It iteratively merges the most frequent pair of adjacent symbols to build a vocabulary.

## Quick start

```julia
include("src/BytePairEncoding.jl")
using .BytePairEncoding

corpus = "low low low lower lower lowest"
tokenizer = train_tokenizer(corpus, 10)

ids = encode(tokenizer, "low lower")
println(ids)                        # [3, 4]
println(decode(tokenizer, ids))     # "low lower"
```

## Usage

### Low-level training

```julia
include("src/BytePairEncoding.jl")
using .BytePairEncoding

corpus = "low low low lower lower lowest"
vocab, merges = train_bpe(corpus, 10)

for (i, merge) in enumerate(merges)
    println("$i. $(merge[1]) + $(merge[2])")
end
```

### Encoding and decoding

```julia
tokens = encode_text("low lower", merges)
println(tokens)                     # ["low</w>", "lower</w>"]
println(decode_tokens(tokens))      # "low lower"
```

### BPETokenizer struct

The `BPETokenizer` bundles merges, vocabulary, and token-to-ID mappings into a single object:

```julia
tokenizer = train_tokenizer(corpus, 10, special_tokens=["<unk>", "<pad>"])

ids = encode(tokenizer, "low lower")    # Vector{Int}
text = decode(tokenizer, ids)           # "low lower"

# Save and reload
save_tokenizer(tokenizer, "my_tokenizer/")
loaded = load_tokenizer("my_tokenizer/")
```

### Token-to-ID mapping

```julia
vocab = get_vocabulary(word_symbols)
extended = add_special_tokens(vocab, ["<unk>", "<pad>"])
index = build_vocab_index(extended, ["<unk>", "<pad>"])

ids = tokens_to_ids(tokens, index)
recovered = ids_to_tokens(ids, index)

save_vocab_index(index, "vocab_index.tsv")
loaded_index = load_vocab_index("vocab_index.tsv")
```

### Sequence utilities

```julia
padded = pad_sequence([1, 2, 3], 5)              # [1, 2, 3, 0, 0]
truncated = truncate_sequence([1, 2, 3, 4, 5], 3) # [1, 2, 3]

batch = prepare_batch([[1,2], [3,4,5,6,7]], 4)
# [[1, 2, 0, 0], [3, 4, 5, 6]]
```

### Regex pre-tokenization

```julia
chunks = pretokenize("Hello, world! It's a test.")
# ["Hello", ",", " world", "!", " It", "'s", " a", " test", "."]

freqs = count_frequencies_pretokenized("hello world hello")

tokens = tokenize("Hello world", merges)  # end-to-end pipeline
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

### Vocabulary analysis

```julia
top = most_common_tokens(tokens, 5)        # top-5 by frequency
avg = average_token_length(vocab)           # mean character count
cov = coverage("low lower xyz", merges)     # fraction of words fully covered
```

### Batch encoding and BPE dropout

```julia
results = encode_batch(["low", "lower", "lowest"], merges)

# Stochastic tokenization for training robustness
tokens = encode_word_with_dropout("lower", merges, dropout=0.1)
```

### Subword regularization

```julia
# N-best tokenizations via BPE dropout
variants = nbest_encode("lower", merges, 5)

# Temperature-controlled sampling
tokens = sample_segmentation("lower", merges, temperature=1.0)
```

### Export / import formats

```julia
# HuggingFace format
export_huggingface_merges(merges, "merges.txt")
loaded = import_huggingface_merges("merges.txt")

# SentencePiece vocab format
export_sentencepiece_vocab(vocab_index, "vocab.model")
```

### Parallel training

```julia
# Run with: julia --threads=4
vocab, merges = train_bpe_parallel(corpus, 100)  # threaded pair counting
```

### Tokenizer comparison

```julia
tokenizers = Dict("bpe" => text -> encode_text(text, merges),
                  "char" => text -> string.(collect(text)))
results = compare_tokenizers("hello world", tokenizers)
ratios = compare_compression("hello world", tokenizers)
```

### Unigram tokenization

```julia
scores = train_unigram(corpus, 50)                     # train vocab with scores
tokens = viterbi_segment("lower", scores)               # optimal segmentation
```

### Configuration files

```julia
config = TokenizerConfig(num_merges=100, min_frequency=2)
save_config(config, "config.json")
loaded = load_config("config.json")
tokenizer = train_from_config(corpus, loaded)
```

### Merge history tracking

```julia
_, merges, history = train_bpe_with_history(corpus, 10)
println(format_merge_history(history))
# Step | Pair            | Freq | New Token   | Vocab Size
# ...
```

### Extended vocabulary analysis

```julia
dist = token_length_distribution(vocab)        # histogram of token lengths
fert = subword_fertility("hello world", merges) # avg tokens per word
overlap = vocab_overlap(vocab1, vocab2)         # Jaccard similarity + set diffs
```

### WordPiece tokenization

```julia
vocab = train_wordpiece(corpus, 30)
tokens = wordpiece_tokenize("lower", vocab)  # ["l", "##o", "##w", "##e", "##r"]
```

### Byte-level BPE

```julia
bytes = text_to_bytes("Hello")              # ["48", "65", "6c", "6c", "6f"]
text = bytes_to_text(bytes)                 # "Hello"

_, byte_merges = train_byte_bpe("low low lower", 5)
tokens = encode_byte_level("low lower", byte_merges)
decoded = bytes_to_text(tokens)
```

### Save and load

```julia
save_merges(merges, "merges.tsv")
loaded_merges = load_merges("merges.tsv")

save_vocab(vocab, "vocab.txt")
```

## Examples

See the `examples/` directory for runnable scripts:

- `examples/basic_training.jl` — train, encode, decode in 20 lines
- `examples/byte_level.jl` — byte-level BPE for language-agnostic tokenization
- `examples/custom_tokenizer.jl` — full workflow with special tokens and save/load
- `examples/multilingual.jl` — training on multilingual text with Unicode normalization

```bash
julia examples/basic_training.jl
```

## CLI scripts

### Train a tokenizer

```bash
julia scripts/train.jl <corpus_file> <num_merges> <output_dir>
julia scripts/train.jl data/sample_corpus.txt 20 /tmp/bpe_out
```

### Encode text

```bash
julia scripts/encode.jl <merges_file> <text>
julia scripts/encode.jl /tmp/bpe_out/merges.tsv "hello world"
```

### Analyze vocabulary

```bash
julia scripts/analyze.jl /tmp/bpe_out "hello world test"
```

### Convert formats

```bash
julia scripts/convert.jl merges.tsv merges.txt hf
```

### Validate tokenizer

```bash
julia scripts/validate.jl /tmp/bpe_out
```

### Interactive playground

```bash
julia scripts/playground.jl
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

### BPETokenizer
- `BPETokenizer` — struct bundling merges, vocab, index, and special tokens
- `train_tokenizer(corpus, num_merges; special_tokens, verbose, min_frequency)` — train a complete tokenizer
- `encode(tokenizer, text)` — tokenize text to integer IDs
- `decode(tokenizer, ids)` — convert IDs back to text
- `save_tokenizer(tokenizer, dir)` — save tokenizer state to directory
- `load_tokenizer(dir)` — load tokenizer from directory

### Encoding and decoding
- `encode_word(word, merges)` — tokenize a single word using learned merges
- `encode_text(text, merges)` — tokenize a full text string
- `decode_tokens(tokens)` — reconstruct text from BPE tokens
- `encode_batch(texts, merges)` — encode multiple texts at once
- `encode_word_with_dropout(word, merges; dropout)` — stochastic tokenization

### Token-to-ID mapping
- `build_vocab_index(vocab, special_tokens)` — assign integer IDs to tokens
- `tokens_to_ids(tokens, index; unk_id)` — map tokens to integer IDs
- `ids_to_tokens(ids, index)` — reverse-map IDs to tokens
- `save_vocab_index(index, path)` — write vocab index to file
- `load_vocab_index(path)` — read vocab index from file

### Sequence utilities
- `pad_sequence(ids, max_len; pad_id)` — right-pad sequence to fixed length
- `truncate_sequence(ids, max_len)` — truncate sequence to max length
- `prepare_batch(batch, max_len; pad_id)` — truncate and pad a batch of sequences

### Pre-tokenization
- `pretokenize(text; pattern)` — regex-based word splitting (GPT-2-style)
- `GPT2_PATTERN` — GPT-2 pre-tokenization regex
- `LLAMA_PATTERN` — LLaMA-style pre-tokenization regex
- `CLIP_PATTERN` — CLIP-style pre-tokenization regex
- `count_frequencies_pretokenized(text; pattern)` — frequency counting with pre-tokenization
- `tokenize(text, merges; pattern)` — end-to-end tokenization pipeline

### Preprocessing and I/O
- `preprocess_text(text; lowercase)` — normalize text for training
- `load_corpus(filepath)` — read a corpus from file
- `save_merges(merges, filepath)` — write merge rules to file
- `load_merges(filepath)` — read merge rules from file
- `save_vocab(vocab, filepath)` — export vocabulary to file

### Analytics and vocabulary analysis
- `compression_ratio(text, tokens)` — characters per token ratio
- `token_frequencies(tokens)` — count token occurrences
- `vocab_size_history(corpus, num_merges)` — track vocab size over training
- `add_special_tokens(vocab, special)` — add special tokens to vocabulary
- `most_common_tokens(tokens, n)` — top-N most frequent tokens
- `average_token_length(vocab)` — mean character count of tokens
- `coverage(text, merges)` — fraction of words fully encodable

### Byte-level BPE
- `text_to_bytes(text)` — convert text to hex byte strings
- `bytes_to_text(byte_tokens)` — reconstruct text from hex byte tokens
- `train_byte_bpe(text, num_merges; verbose)` — train BPE on byte sequences
- `encode_byte_level(text, merges)` — apply byte-level BPE merges

## Running tests

```bash
julia test/runtests.jl
```

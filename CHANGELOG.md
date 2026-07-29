# Changelog

All notable changes to this project will be documented in this file.

## [0.4.0] - 2026-07-29

### Added
- `corpus_statistics()` for quick corpus profiling (word count, unique words, avg length)
- `attention_mask()` generates 1/0 masks from padded token sequences
- `prune_vocabulary()` removes infrequent tokens while preserving single-char tokens
- `tokenizer_summary()` prints a formatted overview of tokenizer state
- `token_length_histogram()` renders a text bar chart of token length distribution
- `save_tokenizer_json` / `load_tokenizer_json` for single-file JSON export/import
- `decode_to_words()` splits token sequences at `</w>` boundaries into words
- `encode_streaming_batch()` encodes files in configurable batch sizes
- `corpus_from_directory()` loads and concatenates text files from a folder
- Left-pad support via `side=:left` kwarg on `pad_sequence`
- NLP pipeline example (`examples/nlp_pipeline.jl`)
- Batch streaming demo in streaming encode example
- Julia 1.11 added to CI test matrix

### Changed
- `load_corpus`, `load_merges`, `load_vocab_index`, `load_tokenizer` now throw `TokenizerError` instead of generic `error()` for consistent error handling
- `count_pairs` and `count_word_frequencies` use `sizehint!` for fewer hash resizes
- `.editorconfig` extended with Makefile and TOML settings
- CONTRIBUTING.md expanded with benchmarking and testing guidance

## [0.3.1] - 2026-07-12

### Added
- Streaming file encoding (`encode_streaming`) with progress callback
- `CachedEncoder` with LRU eviction for repeated word encodings
- Statistics functions: `token_entropy`, `vocabulary_coverage_report`, `oov_rate`
- Input validation for `train_tokenizer`, `train_wordpiece`, `train_unigram`
- Algorithm complexity comments on core functions
- GitHub issue templates for bug reports and feature requests
- Makefile with test, bench, and examples targets
- CHANGELOG.md and CONTRIBUTING.md
- Streaming `--file` mode for encode.jl CLI
- Performance baseline tests and expanded edge case / unicode tests

### Changed
- CI now runs examples as smoke tests
- Export list reorganized into logical groups
- Regex patterns documented with component breakdowns
- README expanded with caching, validation, and statistics API reference

## [0.3.0] - 2026-06-27

### Added
- Parallel BPE training with threaded pair counting (`train_bpe_parallel`)
- HuggingFace and SentencePiece format import/export
- N-best encoding and temperature-controlled sampling for subword regularization
- Performance benchmarks for training, encoding, parallel, byte-level, and memory
- Benchmarks section in README

## [0.2.0] - 2026-06-20

### Added
- WordPiece tokenization (`train_wordpiece`, `wordpiece_tokenize`)
- Unigram tokenization (`train_unigram`, `viterbi_segment`)
- Cross-tokenizer comparison (`compare_tokenizers`, `compare_compression`)
- Merge history tracking (`train_bpe_with_history`, `MergeRecord`)
- Extended vocabulary analysis (`token_length_distribution`, `subword_fertility`, `vocab_overlap`)
- TokenizerConfig for JSON-based configuration
- Integration tests for WordPiece and Unigram pipelines

## [0.1.0] - 2026-06-10

### Added
- Core BPE training algorithm
- Encoding, decoding, and round-trip support
- BPETokenizer struct with save/load
- Token-to-ID mapping and reverse mapping
- Byte-level BPE variant
- Regex pre-tokenization (GPT-2, LLaMA, CLIP patterns)
- BPE dropout and protected token encoding
- Streaming training from file
- Sequence utilities (pad, truncate, batch)
- Vocabulary analytics (compression ratio, coverage, token frequencies)
- CLI scripts for training, encoding, analysis, validation, conversion, comparison
- Interactive playground script
- Example scripts for basic, byte-level, custom, and multilingual usage
- GitHub Actions CI on Julia 1.9, 1.10, and latest

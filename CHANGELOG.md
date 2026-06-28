# Changelog

All notable changes to this project will be documented in this file.

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

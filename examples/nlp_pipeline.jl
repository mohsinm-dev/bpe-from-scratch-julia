#!/usr/bin/env julia
# Example: full NLP preprocessing pipeline
# Train a tokenizer, encode text, generate attention masks, and inspect results.

include(joinpath(@__DIR__, "..", "src", "BytePairEncoding.jl"))
using .BytePairEncoding

# Load and preprocess corpus
corpus_path = joinpath(@__DIR__, "..", "data", "sample_corpus.txt")
corpus = load_corpus(corpus_path)
println("Corpus stats: ", corpus_statistics(corpus))

# Train tokenizer
tokenizer = train_tokenizer(corpus, 20)
println("\n", tokenizer_summary(tokenizer))

# Encode some sentences
sentences = ["low lower lowest", "the new meaning"]
max_len = 8

println("\n--- Encoding pipeline ---")
for sentence in sentences
    ids = encode(tokenizer, sentence)
    padded = pad_sequence(ids, max_len)
    mask = attention_mask(padded)
    println("  \"$sentence\"")
    println("    IDs:    $ids")
    println("    Padded: $padded")
    println("    Mask:   $mask")
end

# Vocabulary histogram
println("\nToken length distribution:")
println(token_length_histogram(tokenizer.vocab, max_width=30))

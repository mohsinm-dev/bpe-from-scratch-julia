#!/usr/bin/env julia

# Basic BPE training example
# Train a tokenizer, encode text, and decode it back

include(joinpath(@__DIR__, "..", "src", "BytePairEncoding.jl"))
using .BytePairEncoding

# training corpus
corpus = "low low low lower lower lowest high higher highest"

# train with 10 merges
vocab, merges = train_bpe(corpus, 10, verbose=true)

# encode new text
text = "low lower highest"
tokens = encode_text(text, merges)
println("\nText: \"$text\"")
println("Tokens: $tokens")

# decode back
decoded = decode_tokens(tokens)
println("Decoded: \"$decoded\"")

# check compression
ratio = compression_ratio(text, tokens)
println("Compression ratio: $(round(ratio, digits=2)) chars/token")

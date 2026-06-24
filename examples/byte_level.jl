#!/usr/bin/env julia

# Byte-level BPE example
# Train and encode at the byte level for language-agnostic tokenization

include(joinpath(@__DIR__, "..", "src", "BytePairEncoding.jl"))
using .BytePairEncoding

corpus = "low low low lower lower lowest café über"

# show byte representation
println("\"Hello\" as bytes: ", text_to_bytes("Hello"))
println("Round-trip: ", bytes_to_text(text_to_bytes("Hello")))

# train byte-level BPE
println("\nTraining byte-level BPE...")
_, byte_merges = train_byte_bpe(corpus, 10, verbose=true)

# encode text
text = "low lower café"
tokens = encode_byte_level(text, byte_merges)
println("\nText: \"$text\"")
println("Byte tokens: $tokens")
println("Decoded: \"$(bytes_to_text(tokens))\"")

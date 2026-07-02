#!/usr/bin/env julia
# Example: streaming file encoding with progress tracking

include(joinpath(@__DIR__, "..", "src", "BytePairEncoding.jl"))
using .BytePairEncoding

# Train on sample corpus
corpus_path = joinpath(@__DIR__, "..", "data", "sample_corpus.txt")
corpus = load_corpus(corpus_path)
_, merges = train_bpe(corpus, 20)

println("Trained $(length(merges)) merges")
println("Streaming encode of: $corpus_path\n")

# Encode with a progress callback
total_tokens = Ref(0)
tokens = encode_streaming(corpus_path, merges, callback=function(line_num, line_tokens)
    total_tokens[] += length(line_tokens)
    println("  line $line_num: $(length(line_tokens)) tokens (total: $(total_tokens[]))")
end)

println("\nDone. $(length(tokens)) tokens total from streaming encode.")
println("First 10 tokens: $(tokens[1:min(10, length(tokens))])")

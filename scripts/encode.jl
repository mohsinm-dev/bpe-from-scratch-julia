#!/usr/bin/env julia

# CLI encoding script for BPE tokenizer
#
# Usage:
#   julia scripts/encode.jl <merges_file> <text>
#
# Example:
#   julia scripts/encode.jl /tmp/bpe_out/merges.tsv "hello world"

include(joinpath(@__DIR__, "..", "src", "BytePairEncoding.jl"))
using .BytePairEncoding

function main()
    if length(ARGS) < 2
        println(stderr, "Usage: julia scripts/encode.jl <merges_file> <text>")
        exit(1)
    end

    merges_file = ARGS[1]
    text = ARGS[2]

    println("Loading merges from: $merges_file")
    merges = load_merges(merges_file)
    println("Loaded $(length(merges)) merges")

    println("\nInput: \"$text\"")

    tokens = encode_text(text, merges)
    println("Tokens: $tokens")
    println("Token count: $(length(tokens))")

    decoded = decode_tokens(tokens)
    println("Decoded: \"$decoded\"")

    ratio = compression_ratio(text, tokens)
    println("Compression ratio: $(round(ratio, digits=2)) chars/token")
end

main()

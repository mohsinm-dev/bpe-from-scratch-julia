#!/usr/bin/env julia

# CLI encoding script for BPE tokenizer
#
# Usage:
#   julia scripts/encode.jl <merges_file> <text>
#   julia scripts/encode.jl <merges_file> --file <input_file>
#
# Example:
#   julia scripts/encode.jl /tmp/bpe_out/merges.tsv "hello world"
#   julia scripts/encode.jl /tmp/bpe_out/merges.tsv --file data/sample_corpus.txt

include(joinpath(@__DIR__, "..", "src", "BytePairEncoding.jl"))
using .BytePairEncoding

function main()
    if "--help" in ARGS || "-h" in ARGS
        println("Encode text using a trained BPE tokenizer.")
        println()
        println("Usage: julia scripts/encode.jl <merges_file> <text>")
        println("       julia scripts/encode.jl <merges_file> --file <input_file>")
        println()
        println("Arguments:")
        println("  merges_file   Path to the merges.tsv file from training")
        println("  text          Text string to tokenize (in quotes)")
        println("  --file        Encode a file line-by-line (streaming mode)")
        println()
        println("Example:")
        println("  julia scripts/encode.jl /tmp/bpe_out/merges.tsv \"hello world\"")
        println("  julia scripts/encode.jl /tmp/bpe_out/merges.tsv --file corpus.txt")
        return
    end

    if length(ARGS) < 2
        println(stderr, "Usage: julia scripts/encode.jl <merges_file> <text|--file path>")
        exit(1)
    end

    merges_file = ARGS[1]
    println("Loading merges from: $merges_file")
    merges = load_merges(merges_file)
    println("Loaded $(length(merges)) merges")

    if ARGS[2] == "--file"
        # streaming mode
        if length(ARGS) < 3
            println(stderr, "Error: --file requires a file path")
            exit(1)
        end
        input_file = ARGS[3]
        println("\nStreaming encode: $input_file")
        tokens = encode_streaming(input_file, merges, callback=function(n, t)
            println("  line $n: $(length(t)) tokens")
        end)
        println("\nTotal tokens: $(length(tokens))")
    else
        # inline text mode
        text = ARGS[2]
        println("\nInput: \"$text\"")

        tokens = encode_text(text, merges)
        println("Tokens: $tokens")
        println("Token count: $(length(tokens))")

        decoded = decode_tokens(tokens)
        println("Decoded: \"$decoded\"")

        ratio = compression_ratio(text, tokens)
        println("Compression ratio: $(round(ratio, digits=2)) chars/token")
    end
end

main()

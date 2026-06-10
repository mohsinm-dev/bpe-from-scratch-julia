#!/usr/bin/env julia

# CLI validation script for BPE tokenizer
#
# Usage:
#   julia scripts/validate.jl <tokenizer_dir>
#
# Example:
#   julia scripts/validate.jl /tmp/bpe_out

include(joinpath(@__DIR__, "..", "src", "BytePairEncoding.jl"))
using .BytePairEncoding

function main()
    if length(ARGS) < 1
        println(stderr, "Usage: julia scripts/validate.jl <tokenizer_dir>")
        exit(1)
    end

    dir = ARGS[1]
    println("Loading tokenizer from: $dir")
    t = load_tokenizer(dir)

    warnings = validate_tokenizer(t)
    if isempty(warnings)
        println("Tokenizer is valid.")
        println("  Merges: $(length(t.merges))")
        println("  Vocab size: $(length(t.vocab))")
        println("  Special tokens: $(t.special_tokens)")
    else
        println("Tokenizer has $(length(warnings)) issue(s):")
        for w in warnings
            println("  - $w")
        end
        exit(1)
    end
end

main()

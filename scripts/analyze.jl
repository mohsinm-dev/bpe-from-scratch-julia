#!/usr/bin/env julia

# CLI vocabulary analysis tool
#
# Usage:
#   julia scripts/analyze.jl <tokenizer_dir> [text]
#
# Example:
#   julia scripts/analyze.jl /tmp/bpe_out "hello world this is a test"

include(joinpath(@__DIR__, "..", "src", "BytePairEncoding.jl"))
using .BytePairEncoding

function main()
    if length(ARGS) < 1
        println(stderr, "Usage: julia scripts/analyze.jl <tokenizer_dir> [text]")
        exit(1)
    end

    dir = ARGS[1]
    println("Loading tokenizer from: $dir")
    t = load_tokenizer(dir)

    println("\n--- Tokenizer Summary ---")
    println("  Merges:         $(length(t.merges))")
    println("  Vocab size:     $(length(t.vocab))")
    println("  Special tokens: $(t.special_tokens)")
    println("  Avg token len:  $(round(average_token_length(t.vocab), digits=2))")

    dist = token_length_distribution(t.vocab)
    println("\n--- Token Length Distribution ---")
    for len in sort(collect(keys(dist)))
        bar = "█" ^ min(dist[len], 40)
        println("  $(lpad(len, 3)) chars: $(rpad(bar, 40)) $(dist[len])")
    end

    if length(ARGS) >= 2
        text = ARGS[2]
        println("\n--- Encoding Analysis ---")
        ids = encode(t, text)
        println("  Input:       \"$text\"")
        println("  Token IDs:   $ids")
        println("  Token count: $(length(ids))")
        println("  Decoded:     \"$(decode(t, ids))\"")

        tokens = encode_text(text, t.merges)
        ratio = compression_ratio(text, tokens)
        println("  Compression: $(round(ratio, digits=2)) chars/token")

        fert = subword_fertility(text, t.merges)
        println("  Fertility:   $(round(fert, digits=2)) tokens/word")

        cov = coverage(text, t.merges)
        println("  Coverage:    $(round(cov * 100, digits=1))%")
    end
end

main()

#!/usr/bin/env julia

# Benchmark byte-level BPE vs character-level BPE

include(joinpath(@__DIR__, "..", "src", "BytePairEncoding.jl"))
using .BytePairEncoding

function benchmark_byte_level()
    println("=== Byte-Level vs Character-Level BPE Benchmark ===\n")

    corpus = "low low low lower lower lowest high higher highest run running runner " ^ 50

    for num_merges in [10, 25, 50]
        # character-level
        t_char = @elapsed begin
            _, merges_char = train_bpe(corpus, num_merges)
        end
        tokens_char = encode_text("low lower highest running", merges_char)

        # byte-level
        t_byte = @elapsed begin
            _, merges_byte = train_byte_bpe(corpus, num_merges)
        end
        tokens_byte = encode_byte_level("low lower highest running", merges_byte)

        println("Merges: $num_merges")
        println("  Char-level:  $(round(t_char * 1000, digits=2)) ms, $(length(tokens_char)) tokens")
        println("  Byte-level:  $(round(t_byte * 1000, digits=2)) ms, $(length(tokens_byte)) tokens")
        println()
    end
end

benchmark_byte_level()

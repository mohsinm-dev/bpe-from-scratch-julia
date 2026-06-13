#!/usr/bin/env julia

# Benchmark memory usage across corpus sizes

include(joinpath(@__DIR__, "..", "src", "BytePairEncoding.jl"))
using .BytePairEncoding

function build_corpus(n_words::Int)::String
    words = ["low", "lower", "lowest", "high", "higher", "highest",
             "run", "running", "runner", "the", "cat", "sat", "on"]
    return join(rand(words, n_words), " ")
end

function benchmark_memory()
    println("=== Memory Usage Benchmark ===\n")
    println("Corpus (words) | Merges | Peak Alloc (MB)")
    println("-" ^ 45)

    for n_words in [100, 1000, 5000, 10000]
        corpus = build_corpus(n_words)
        for num_merges in [10, 50]
            # warmup
            train_bpe(build_corpus(10), 1)
            alloc = @allocated train_bpe(corpus, num_merges)
            println("$(lpad(n_words, 14)) | $(lpad(num_merges, 6)) | $(lpad(round(alloc / 1e6, digits=2), 15))")
        end
    end
end

benchmark_memory()

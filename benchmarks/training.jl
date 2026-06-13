#!/usr/bin/env julia

# Benchmark training performance at various merge counts

include(joinpath(@__DIR__, "..", "src", "BytePairEncoding.jl"))
using .BytePairEncoding

function build_corpus(n_words::Int)::String
    words = ["low", "lower", "lowest", "high", "higher", "highest",
             "run", "running", "runner", "runs", "play", "playing",
             "the", "cat", "sat", "on", "mat", "dog", "fast", "faster"]
    return join(rand(words, n_words), " ")
end

function benchmark_training()
    println("=== Training Benchmark ===\n")
    println("Corpus size (words) | Merges | Time (ms) | Alloc (MB)")
    println("-" ^ 55)

    for n_words in [100, 500, 1000, 5000]
        corpus = build_corpus(n_words)
        for num_merges in [10, 50, 100]
            # warmup
            train_bpe(corpus, 1)

            t = @elapsed begin
                alloc = @allocated train_bpe(corpus, num_merges)
            end
            t2 = @elapsed train_bpe(corpus, num_merges)

            println("$(lpad(n_words, 19)) | $(lpad(num_merges, 6)) | $(lpad(round(t2 * 1000, digits=1), 9)) | $(lpad(round(alloc / 1e6, digits=2), 9))")
        end
    end
end

benchmark_training()

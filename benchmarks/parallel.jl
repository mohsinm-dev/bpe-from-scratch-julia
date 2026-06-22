#!/usr/bin/env julia

# Benchmark parallel vs sequential training
# Run with: julia --threads=4 benchmarks/parallel.jl

include(joinpath(@__DIR__, "..", "src", "BytePairEncoding.jl"))
using .BytePairEncoding

function build_corpus(n_words::Int)::String
    words = ["low", "lower", "lowest", "high", "higher", "highest",
             "run", "running", "runner", "runs", "play", "playing",
             "the", "cat", "sat", "on", "mat", "dog", "fast", "faster"]
    return join(rand(words, n_words), " ")
end

function benchmark_parallel()
    println("=== Parallel vs Sequential Training ===")
    println("Threads available: $(Threads.nthreads())\n")

    println("Corpus (words) | Merges | Sequential (ms) | Parallel (ms) | Speedup")
    println("-" ^ 70)

    for n_words in [500, 2000, 5000]
        corpus = build_corpus(n_words)
        for num_merges in [20, 50]
            # warmup
            train_bpe(corpus, 1)
            train_bpe_parallel(corpus, 1)

            t_seq = @elapsed train_bpe(corpus, num_merges)
            t_par = @elapsed train_bpe_parallel(corpus, num_merges)
            speedup = t_seq / max(t_par, 1e-9)

            println("$(lpad(n_words, 14)) | $(lpad(num_merges, 6)) | $(lpad(round(t_seq*1000, digits=1), 15)) | $(lpad(round(t_par*1000, digits=1), 13)) | $(lpad(round(speedup, digits=2), 7))x")
        end
    end
end

benchmark_parallel()

#!/usr/bin/env julia

# Benchmark encoding throughput

include(joinpath(@__DIR__, "..", "src", "BytePairEncoding.jl"))
using .BytePairEncoding

function benchmark_encoding()
    println("=== Encoding Benchmark ===\n")

    corpus = "low low low lower lower lowest high higher highest run running runner " ^50
    _, merges = train_bpe(corpus, 50)

    texts = [
        "low lower lowest",
        "the cat sat on the mat " ^ 10,
        "running runner runs " ^ 20,
        corpus,
    ]

    println("Text length (chars) | Tokens | Time (μs) | Tokens/sec")
    println("-" ^ 55)

    for text in texts
        # warmup
        encode_text(text, merges)

        n_runs = 100
        t = @elapsed for _ in 1:n_runs
            encode_text(text, merges)
        end
        avg_time = t / n_runs
        tokens = encode_text(text, merges)
        tps = length(tokens) / avg_time

        println("$(lpad(length(text), 19)) | $(lpad(length(tokens), 6)) | $(lpad(round(avg_time * 1e6, digits=1), 9)) | $(lpad(round(tps, digits=0), 10))")
    end
end

benchmark_encoding()

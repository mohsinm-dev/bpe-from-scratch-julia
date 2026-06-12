#!/usr/bin/env julia

# Generate a synthetic corpus for benchmarking
#
# Usage:
#   julia scripts/generate_corpus.jl <output_file> <num_lines>
#
# Example:
#   julia scripts/generate_corpus.jl /tmp/large_corpus.txt 10000

function main()
    if length(ARGS) < 2
        println(stderr, "Usage: julia scripts/generate_corpus.jl <output_file> <num_lines>")
        exit(1)
    end

    output_file = ARGS[1]
    num_lines = parse(Int, ARGS[2])

    words = [
        "the", "a", "is", "of", "and", "to", "in", "that", "it", "was",
        "for", "on", "are", "with", "as", "at", "be", "this", "have", "from",
        "natural", "language", "processing", "machine", "learning", "deep",
        "neural", "network", "model", "training", "tokenization", "encoding",
        "byte", "pair", "subword", "vocabulary", "merge", "frequency",
        "algorithm", "data", "text", "word", "token", "sequence", "input",
        "running", "runner", "runs", "played", "playing", "player",
        "lower", "lowest", "low", "higher", "highest", "high",
        "faster", "fastest", "fast", "slower", "slowest", "slow",
    ]

    open(output_file, "w") do io
        for _ in 1:num_lines
            n = rand(5:15)
            line = join(rand(words, n), " ")
            println(io, line)
        end
    end

    println("Generated $num_lines lines to $output_file")
end

main()

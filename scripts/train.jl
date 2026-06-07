#!/usr/bin/env julia

# CLI training script for BPE tokenizer
#
# Usage:
#   julia scripts/train.jl <corpus_file> <num_merges> <output_dir>
#
# Example:
#   julia scripts/train.jl data/sample_corpus.txt 20 /tmp/bpe_out

include(joinpath(@__DIR__, "..", "src", "BytePairEncoding.jl"))
using .BytePairEncoding

function main()
    if length(ARGS) < 3
        println(stderr, "Usage: julia scripts/train.jl <corpus_file> <num_merges> <output_dir>")
        exit(1)
    end

    corpus_file = ARGS[1]
    num_merges = parse(Int, ARGS[2])
    output_dir = ARGS[3]

    println("Loading corpus from: $corpus_file")
    corpus = load_corpus(corpus_file)
    println("Corpus size: $(length(corpus)) characters")

    println("Training BPE with $num_merges merges...")
    tokenizer = train_tokenizer(corpus, num_merges, verbose=true)

    println("\nSaving tokenizer to: $output_dir")
    save_tokenizer(tokenizer, output_dir)

    println("\nTraining complete:")
    println("  Merges: $(length(tokenizer.merges))")
    println("  Vocab size: $(length(tokenizer.vocab))")
    println("  Output files:")
    println("    $(joinpath(output_dir, "merges.tsv"))")
    println("    $(joinpath(output_dir, "vocab_index.tsv"))")
    println("    $(joinpath(output_dir, "special_tokens.txt"))")
end

main()

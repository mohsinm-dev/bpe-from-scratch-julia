#!/usr/bin/env julia

# Compare BPE, WordPiece, and Unigram tokenizers on the same text

include(joinpath(@__DIR__, "..", "src", "BytePairEncoding.jl"))
using .BytePairEncoding

function main()
    corpus = "low low low lower lower lowest high higher highest run running runner the cat sat on the mat"

    println("=== Training tokenizers ===\n")

    # BPE
    _, bpe_merges = train_bpe(corpus, 20)
    println("BPE: $(length(bpe_merges)) merges")

    # WordPiece
    wp_vocab = train_wordpiece(corpus, 40)
    println("WordPiece: $(length(wp_vocab)) vocab")

    # Unigram
    uni_scores = train_unigram(corpus, 25)
    println("Unigram: $(length(uni_scores)) vocab")

    println("\n=== Tokenization comparison ===\n")

    test_texts = ["low lower", "running runner", "the cat sat"]

    for text in test_texts
        println("Text: \"$text\"")
        bpe_tokens = encode_text(text, bpe_merges)
        println("  BPE:       $bpe_tokens ($(length(bpe_tokens)) tokens)")

        wp_tokens = String[]
        for word in split(text)
            append!(wp_tokens, wordpiece_tokenize(String(word), wp_vocab))
        end
        println("  WordPiece: $wp_tokens ($(length(wp_tokens)) tokens)")

        uni_tokens = String[]
        for word in split(text)
            append!(uni_tokens, viterbi_segment(String(word), uni_scores))
        end
        println("  Unigram:   $uni_tokens ($(length(uni_tokens)) tokens)")
        println()
    end
end

main()

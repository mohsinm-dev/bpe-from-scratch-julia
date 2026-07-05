#!/usr/bin/env julia
# Example: compare BPE, WordPiece, and Unigram on the same corpus

include(joinpath(@__DIR__, "..", "src", "BytePairEncoding.jl"))
using .BytePairEncoding

corpus = """
the quick brown fox jumps over the lazy dog
the fox ran quickly across the field
a brown dog chased the quick fox
the lazy dog slept under the tree
"""

println("=== Tokenizer Comparison ===\n")

# Train all three
_, bpe_merges = train_bpe(corpus, 20)
wp_vocab = train_wordpiece(corpus, 30)
uni_scores = train_unigram(corpus, 25)

test_words = ["quickly", "chased", "brown", "jumps"]

for word in test_words
    bpe_tokens = encode_word(word, bpe_merges)
    wp_tokens = wordpiece_tokenize(word, wp_vocab)
    uni_tokens = viterbi_segment(word, uni_scores)

    println("\"$word\":")
    println("  BPE:       $(join(bpe_tokens, " | "))")
    println("  WordPiece: $(join(wp_tokens, " | "))")
    println("  Unigram:   $(join(uni_tokens, " | "))")
    println()
end

# Compression comparison
text = "the quick brown fox jumps over the lazy dog"
bpe_tokens = encode_text(text, bpe_merges)
bpe_ratio = compression_ratio(text, bpe_tokens)
println("Compression (BPE): $(round(bpe_ratio, digits=2)) chars/token")
println("Fertility (BPE):   $(round(subword_fertility(text, bpe_merges), digits=2)) tokens/word")

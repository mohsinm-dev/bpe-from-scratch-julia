#!/usr/bin/env julia

# Multilingual tokenization example

include(joinpath(@__DIR__, "..", "src", "BytePairEncoding.jl"))
using .BytePairEncoding

# load multilingual corpus
corpus = load_corpus(joinpath(@__DIR__, "..", "data", "multilingual_corpus.txt"))
println("Corpus preview: ", first(corpus, 80), "...")

# normalize unicode before training
normalized = normalize_unicode(corpus)

# train tokenizer
tokenizer = train_tokenizer(normalized, 30)
println("\nVocab size: $(length(tokenizer.vocab))")

# test on different languages
texts = [
    "the cat sat",
    "café résumé",
    "自然语言",
]

for text in texts
    ids = encode(tokenizer, text)
    decoded = decode(tokenizer, ids)
    println("\n\"$text\" → $ids → \"$decoded\"")
end

# measure coverage
for text in texts
    cov = coverage(text, tokenizer.merges)
    println("Coverage of \"$text\": $(round(cov * 100, digits=1))%")
end

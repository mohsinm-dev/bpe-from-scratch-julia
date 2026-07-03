#!/usr/bin/env julia
# Benchmark: CachedEncoder vs raw encode_word

include(joinpath(@__DIR__, "..", "src", "BytePairEncoding.jl"))
using .BytePairEncoding

println("=== Cache Benchmark ===\n")

corpus_path = joinpath(@__DIR__, "..", "data", "sample_corpus.txt")
corpus = load_corpus(corpus_path)
_, merges = train_bpe(corpus, 50)

# Build a realistic word list with repetition (Zipf-like)
words_raw = split(corpus)
words = String[]
for w in words_raw
    append!(words, fill(String(w), rand(1:5)))
end

println("Words to encode: $(length(words))")
println("Unique words:    $(length(Set(words)))")
println("Merges:          $(length(merges))")

# Uncached
t_uncached = @elapsed begin
    for w in words
        encode_word(w, merges)
    end
end
println("\nUncached:  $(round(t_uncached * 1000, digits=2)) ms")

# Cached
enc = CachedEncoder(merges)
t_cached = @elapsed begin
    for w in words
        cached_encode_word(enc, w)
    end
end
stats = cache_stats(enc)
println("Cached:    $(round(t_cached * 1000, digits=2)) ms")
println("Hit rate:  $(round(stats.hit_rate * 100, digits=1))%")
println("Speedup:   $(round(t_uncached / t_cached, digits=1))x")

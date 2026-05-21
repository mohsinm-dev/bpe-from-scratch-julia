include("../src/BytePairEncoding.jl")

using .BytePairEncoding

corpus = "low low low lower lower lowest"

println("=== Step by step ===\n")

frequencies = count_word_frequencies(corpus)
println("Word frequencies:")
for (word, count) in sort(collect(frequencies), by=x -> -x[2])
    println("  $word => $count")
end

word_symbols = initialize_word_symbols(frequencies)
println("\nInitial symbols:")
for (symbols, freq) in word_symbols
    println("  ", join(symbols, " "), " => $freq")
end

pair_counts = count_pairs(word_symbols)
println("\nPair counts:")
for (pair, count) in sort(collect(pair_counts), by=x -> -x[2])
    println("  $(pair[1]) $(pair[2]) => $count")
end

println("\nBest pair: ", best_pair(pair_counts))

println("\n=== Full training (10 merges) ===\n")

vocab, merges = train_bpe(corpus, 10)

println("Merges performed:")
for (i, merge) in enumerate(merges)
    println("  $i. $(merge[1]) + $(merge[2]) -> $(merge[1])$(merge[2])")
end

println("\nFinal vocabulary:")
for (symbols, freq) in vocab
    println("  ", join(symbols, " "), " => $freq")
end

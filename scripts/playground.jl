include("../src/BytePairEncoding.jl")

using .BytePairEncoding

corpus = "low lower lowest lower"

frequencies = count_word_frequencies(corpus)
word_symbols = initialize_word_symbols(frequencies)
pair_counts = count_pairs(word_symbols)
pair = best_pair(pair_counts)

println("Word symbols:")
println(word_symbols)

println("\nPair counts:")
for (pair, count) in pair_counts
    println(pair, " => ", count)
end


println("\n Best Pairs:")
println(pair)

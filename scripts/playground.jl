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

println("\n=== Full training (10 merges, verbose) ===\n")

vocab, merges = train_bpe(corpus, 10, verbose=true)

println("\nFinal vocabulary:")
for (symbols, freq) in vocab
    println("  ", join(symbols, " "), " => $freq")
end

println("\n=== Encoding new text ===\n")

test_text = "low lower"
tokens = encode_text(test_text, merges)
println("Text: \"$test_text\"")
println("Tokens: $tokens")
println("Decoded: \"$(decode_tokens(tokens))\"")

ratio = compression_ratio(test_text, tokens)
println("Compression ratio: $(round(ratio, digits=2)) chars/token")

println("\n=== Token frequency analysis ===\n")

freqs = token_frequencies(tokens)
for (token, count) in sort(collect(freqs), by=x -> -x[2])
    println("  $token => $count")
end

println("\n=== Batch encoding ===\n")

texts = ["low", "lower", "lowest"]
batch_results = encode_batch(texts, merges)
for (text, toks) in zip(texts, batch_results)
    println("  \"$text\" => $toks")
end

println("\n=== Save and reload merges ===\n")

tmpfile = tempname()
save_merges(merges, tmpfile)
reloaded = load_merges(tmpfile)
println("Saved $(length(merges)) merges, loaded $(length(reloaded)) merges")
println("Round-trip OK: $(merges == reloaded)")
rm(tmpfile)

println("\n=== Vocab size history ===\n")

history = vocab_size_history(corpus, 10)
for (step, size) in enumerate(history)
    label = step == 1 ? "initial" : "merge $(step - 1)"
    println("  $label: $size tokens")
end

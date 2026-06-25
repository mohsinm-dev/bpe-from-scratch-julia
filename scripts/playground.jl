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

println("\n=== BPETokenizer struct ===\n")

tokenizer = train_tokenizer(corpus, 10)
println("Trained tokenizer:")
println("  Merges: $(length(tokenizer.merges))")
println("  Vocab size: $(length(tokenizer.vocab))")
println("  Special tokens: $(tokenizer.special_tokens)")

ids = encode(tokenizer, "low lower")
println("\nEncode \"low lower\": $ids")
println("Decode back: \"$(decode(tokenizer, ids))\"")

# save and reload tokenizer
dir = mktempdir()
save_tokenizer(tokenizer, dir)
reloaded_tokenizer = load_tokenizer(dir)
println("\nSaved and reloaded tokenizer")
println("Re-encode matches: $(encode(reloaded_tokenizer, "low lower") == ids)")
rm(dir, recursive=true)

println("\n=== Sequence utilities ===\n")

println("Pad [1,2,3] to length 5: $(pad_sequence([1,2,3], 5))")
println("Truncate [1,2,3,4,5] to length 3: $(truncate_sequence([1,2,3,4,5], 3))")
batch = prepare_batch([[1,2], [3,4,5,6,7], [8]], 4)
println("Prepare batch to length 4: $batch")

println("\n=== Pretokenization ===\n")

sample = "Hello, world! It's a test."
chunks = pretokenize(sample)
println("Pretokenize \"$sample\":")
for chunk in chunks
    println("  \"$chunk\"")
end

println("\n=== Vocabulary analysis ===\n")

all_tokens = encode_text("low low lower lower lowest", merges)
top = most_common_tokens(all_tokens, 3)
println("Top 3 tokens:")
for (token, count) in top
    println("  $token => $count")
end

v = get_vocabulary(vocab)
println("\nAverage token length: $(round(average_token_length(v), digits=2))")
println("Coverage on trained words: $(coverage("low lower lowest", merges))")
println("Coverage with unknown word: $(coverage("low xyz", merges))")

println("\n=== Subword regularization ===\n")

variants = nbest_encode("lowest", merges, 5)
println("N-best encodings of \"lowest\":")
for (i, v) in enumerate(variants)
    println("  $i. $v")
end

println("\nSampled segmentations at different temperatures:")
for temp in [0.0, 1.0, 3.0]
    s = sample_segmentation("lowest", merges, temperature=temp)
    println("  temp=$temp: $s")
end

println("\n=== Pretokenization patterns ===\n")

sample_pt = "Hello, world! It's a test 123."
println("GPT-2:  ", pretokenize(sample_pt))
println("LLaMA:  ", pretokenize(sample_pt, pattern=LLAMA_PATTERN))
println("CLIP:   ", pretokenize(sample_pt, pattern=CLIP_PATTERN))

println("\n=== Unigram tokenization ===\n")

scores = train_unigram(corpus, 20)
println("Unigram vocab size: $(length(scores))")
for word in ["low", "lower", "lowest"]
    tokens_u = viterbi_segment(word, scores)
    println("  \"$word\" => $tokens_u")
end

println("\n=== Merge history ===\n")

_, _, history = train_bpe_with_history(corpus, 10)
println(format_merge_history(history))

println("\n=== Vocabulary analysis (extended) ===\n")

dist = token_length_distribution(v)
println("Token length distribution:")
for len in sort(collect(keys(dist)))
    println("  length $len: $(dist[len]) tokens")
end

fert = subword_fertility("low lower lowest running", merges)
println("\nSubword fertility: $(round(fert, digits=2)) tokens/word")

println("\n=== WordPiece tokenization ===\n")

wp_vocab = train_wordpiece(corpus, 30)
println("WordPiece vocab size: $(length(wp_vocab))")
for word in ["low", "lower", "lowest"]
    wp_tokens = wordpiece_tokenize(word, wp_vocab)
    println("  \"$word\" => $wp_tokens")
end

println("\n=== Byte-level BPE ===\n")

println("\"Low\" as bytes: $(text_to_bytes("Low"))")
println("Bytes back to text: $(bytes_to_text(["4c", "6f", "77"]))")

_, byte_merges = train_byte_bpe(corpus, 5, verbose=true)
byte_tokens = encode_byte_level("low lower", byte_merges)
println("\nByte-level tokens: $byte_tokens")
println("Decoded: $(bytes_to_text(byte_tokens))")

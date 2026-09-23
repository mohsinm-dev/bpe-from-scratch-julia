include("../src/BytePairEncoding.jl")
using .BytePairEncoding

corpus = "the quick brown fox jumps over the lazy dog the fox the dog"
println("=== Corpus Statistics ===")
stats = corpus_statistics(corpus)
println("  Words: $(stats.word_count), Unique: $(stats.unique_words)")

tokenizer = train_tokenizer(corpus, 15)
println("\n=== Tokenizer ===")
println(tokenizer_summary(tokenizer))
println("  Merge depth: $(merge_depth(tokenizer.merges))")
println("  Vocab size: $(vocab_size(tokenizer))")

test_texts = ["the fox", "quick dog", "xyz unknown"]
covs = batch_coverage(test_texts, tokenizer.merges)
println("\n=== Coverage ===")
for (text, cov) in zip(test_texts, covs)
    println("  \"$text\" → $(round(cov * 100, digits=1))%")
end

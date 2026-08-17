# Encoding throughput benchmark
# Measures tokens/second for various corpus sizes

include("../src/BytePairEncoding.jl")
using .BytePairEncoding

words = ["the", "quick", "brown", "fox", "jumps", "over", "lazy", "dog",
         "running", "runner", "highest", "lower", "lowest", "newer"]

for size in [100, 500, 1000, 5000]
    corpus = join(rand(words, size), " ")
    _, merges = train_bpe(corpus, 50)

    t_start = time()
    tokens = encode_text(corpus, merges)
    elapsed = time() - t_start

    throughput = length(tokens) / elapsed
    println("corpus=$size words | $(length(tokens)) tokens | $(round(throughput, digits=0)) tokens/sec")
end

# Batch encoding benchmark
# Compares sequential vs parallel batch encoding

include("../src/BytePairEncoding.jl")
using .BytePairEncoding

words = ["the", "quick", "brown", "fox", "jumps", "over", "lazy", "dog"]
corpus = join(rand(words, 2000), " ")
_, merges = train_bpe(corpus, 50)

texts = [join(rand(words, 20), " ") for _ in 1:100]

t_seq = @elapsed begin
    seq_results = encode_batch(texts, merges)
end

t_par = @elapsed begin
    par_results = parallel_encode_batch(texts, merges)
end

println("Sequential: $(round(t_seq * 1000, digits=1))ms")
println("Parallel:   $(round(t_par * 1000, digits=1))ms")
println("Speedup:    $(round(t_seq / t_par, digits=2))x")

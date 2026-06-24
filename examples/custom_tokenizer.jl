#!/usr/bin/env julia

# Full tokenizer workflow with special tokens, save/load

include(joinpath(@__DIR__, "..", "src", "BytePairEncoding.jl"))
using .BytePairEncoding

corpus = "the cat sat on the mat the dog sat on the log running runner runs"

# train with custom special tokens
tokenizer = train_tokenizer(corpus, 15,
    special_tokens=["<unk>", "<pad>", "<bos>", "<eos>"])

println("Vocab size: $(length(tokenizer.vocab))")
println("Special tokens: $(tokenizer.special_tokens)")

# encode to IDs
text = "the cat runs"
ids = encode(tokenizer, text)
println("\nText: \"$text\"")
println("IDs: $ids")
println("Decoded: \"$(decode(tokenizer, ids))\"")

# save and reload
dir = mktempdir()
save_tokenizer(tokenizer, dir)
println("\nSaved to: $dir")

loaded = load_tokenizer(dir)
@assert encode(loaded, text) == ids "round-trip failed"
println("Reload round-trip OK")

# validate
warnings = validate_tokenizer(loaded)
println("Validation: $(isempty(warnings) ? "passed" : warnings)")

rm(dir, recursive=true)

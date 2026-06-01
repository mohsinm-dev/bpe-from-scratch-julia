using Test

include("../src/BytePairEncoding.jl")
using .BytePairEncoding

@testset "preprocess_text" begin
    @test preprocess_text("  Hello   World  ") == "hello world"
    @test preprocess_text("UPPER", lowercase=true) == "upper"
    @test preprocess_text("KEEP", lowercase=false) == "KEEP"
    @test preprocess_text("a\t\nb") == "a b"
end

@testset "load_corpus" begin
    path = joinpath(@__DIR__, "..", "data", "sample_corpus.txt")
    corpus = load_corpus(path)
    @test length(corpus) > 0
    @test occursin("low", corpus)
    @test_throws ErrorException load_corpus("nonexistent_file.txt")
end

@testset "word_to_symbols" begin
    @test word_to_symbols("low") == ["l", "o", "w", "</w>"]
    @test word_to_symbols("a") == ["a", "</w>"]
    @test word_to_symbols("hi") == ["h", "i", "</w>"]
end

@testset "count_word_frequencies" begin
    freqs = count_word_frequencies("low lower lowest lower")
    @test freqs["low"] == 1
    @test freqs["lower"] == 2
    @test freqs["lowest"] == 1
    @test length(freqs) == 3
end

@testset "initialize_word_symbols" begin
    freqs = Dict("low" => 2)
    ws = initialize_word_symbols(freqs)
    @test ws[["l", "o", "w", "</w>"]] == 2
end

@testset "count_pairs" begin
    ws = Dict(["l", "o", "w", "</w>"] => 3)
    pc = count_pairs(ws)
    @test pc[("l", "o")] == 3
    @test pc[("o", "w")] == 3
    @test pc[("w", "</w>")] == 3
end

@testset "best_pair" begin
    pc = Dict(("a", "b") => 5, ("c", "d") => 10)
    @test best_pair(pc) == ("c", "d")
    @test best_pair(Dict{Tuple{String,String},Int}()) === nothing
end

@testset "merge_symbols" begin
    symbols = ["l", "o", "w", "</w>"]
    merged = merge_symbols(symbols, ("l", "o"))
    @test merged == ["lo", "w", "</w>"]

    symbols2 = ["a", "b", "a", "b", "c"]
    merged2 = merge_symbols(symbols2, ("a", "b"))
    @test merged2 == ["ab", "ab", "c"]
end

@testset "train_bpe" begin
    corpus = "low low low lower lower lowest"
    vocab, merges = train_bpe(corpus, 3)
    @test length(merges) == 3
    @test merges[1] isa Tuple{String,String}
end

@testset "train_bpe verbose and min_frequency" begin
    corpus = "low low low lower lower lowest"

    # verbose mode should not error (output goes to stdout)
    vocab_v, merges_v = train_bpe(corpus, 3, verbose=true)
    @test length(merges_v) == 3

    # min_frequency should stop early when pair freq drops below threshold
    _, merges_mf = train_bpe(corpus, 100, min_frequency=100)
    @test length(merges_mf) < 100

    # combined: verbose + min_frequency
    _, merges_both = train_bpe(corpus, 100, verbose=true, min_frequency=100)
    @test length(merges_both) == length(merges_mf)
end

@testset "encode_word" begin
    merges = [("l", "o"), ("lo", "w")]
    @test encode_word("low", merges) == ["low", "</w>"]
    @test encode_word("a", Tuple{String,String}[]) == ["a", "</w>"]
    @test encode_word("lo", merges) == ["lo", "</w>"]
end

@testset "encode_text" begin
    merges = [("l", "o"), ("lo", "w")]
    tokens = encode_text("low lo", merges)
    @test tokens == ["low", "</w>", "lo", "</w>"]
    @test encode_text("a", Tuple{String,String}[]) == ["a", "</w>"]
end

@testset "decode_tokens" begin
    @test decode_tokens(["low", "</w>", "er", "</w>"]) == "low er"
    @test decode_tokens(["h", "i", "</w>"]) == "hi"
    @test decode_tokens(String[]) == ""
end

@testset "save_merges and load_merges" begin
    merges = [("l", "o"), ("lo", "w"), ("low", "</w>")]
    tmpfile = tempname()
    try
        save_merges(merges, tmpfile)
        loaded = load_merges(tmpfile)
        @test loaded == merges
        @test length(loaded) == 3
    finally
        isfile(tmpfile) && rm(tmpfile)
    end
    @test_throws ErrorException load_merges("nonexistent_merges.txt")
end

@testset "save_vocab" begin
    vocab = Set(["lo", "w", "</w>", "er"])
    tmpfile = tempname()
    try
        save_vocab(vocab, tmpfile)
        lines = readlines(tmpfile)
        @test length(lines) == 4
        @test lines == sort(collect(vocab))
    finally
        isfile(tmpfile) && rm(tmpfile)
    end
end

@testset "compression_ratio" begin
    @test compression_ratio("hello world", ["hel", "lo", "</w>", "wor", "ld", "</w>"]) == 11 / 6
    @test compression_ratio("hi", ["h", "i", "</w>"]) ≈ 2 / 3
    @test compression_ratio("test", String[]) == 0.0
end

@testset "token_frequencies" begin
    tokens = ["lo", "w", "</w>", "lo", "w", "er", "</w>"]
    freqs = token_frequencies(tokens)
    @test freqs["lo"] == 2
    @test freqs["w"] == 2
    @test freqs["</w>"] == 2
    @test freqs["er"] == 1
end

@testset "vocab_size_history" begin
    corpus = "low low low lower lower lowest"
    history = vocab_size_history(corpus, 5)
    @test length(history) >= 2
    @test history[1] > 0
    @test history isa Vector{Int}
end

@testset "add_special_tokens" begin
    vocab = Set(["lo", "w", "</w>"])
    extended = add_special_tokens(vocab, ["<unk>", "<pad>"])
    @test "<unk>" in extended
    @test "<pad>" in extended
    @test "lo" in extended
    @test length(extended) == 5
    # original unchanged
    @test length(vocab) == 3
end

@testset "encode_batch" begin
    merges = [("l", "o")]
    results = encode_batch(["lo", "la"], merges)
    @test length(results) == 2
    @test results[1] == ["lo", "</w>"]
    @test results[2] == ["l", "a", "</w>"]
end

@testset "encode_word_with_dropout" begin
    merges = [("l", "o"), ("lo", "w")]
    # with zero dropout, identical to encode_word
    @test encode_word_with_dropout("low", merges, dropout=0.0) == ["low", "</w>"]
    # with full dropout, should return character-level tokenization
    @test encode_word_with_dropout("low", merges, dropout=1.0) == ["l", "o", "w", "</w>"]
end

@testset "get_vocabulary" begin
    ws = Dict(["lo", "w", "</w>"] => 3, ["lo", "w", "er", "</w>"] => 2)
    v = get_vocabulary(ws)
    @test "lo" in v
    @test "w" in v
    @test "</w>" in v
    @test "er" in v
    @test length(v) == 4
end

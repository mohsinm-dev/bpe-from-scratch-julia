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

@testset "integration: train, encode, decode, save, load" begin
    # load corpus from file
    path = joinpath(@__DIR__, "..", "data", "sample_corpus.txt")
    raw_corpus = load_corpus(path)
    corpus = preprocess_text(raw_corpus)

    # train
    vocab, merges = train_bpe(corpus, 20)
    @test length(merges) > 0
    @test length(merges) <= 20

    # encode and decode round-trip
    test_text = "low lower lowest"
    tokens = encode_text(test_text, merges)
    @test length(tokens) > 0
    decoded = decode_tokens(tokens)
    @test decoded == test_text

    # save and reload merges
    tmpfile = tempname()
    try
        save_merges(merges, tmpfile)
        reloaded = load_merges(tmpfile)
        @test reloaded == merges
        # encoding with reloaded merges should give same result
        @test encode_text(test_text, reloaded) == tokens
    finally
        isfile(tmpfile) && rm(tmpfile)
    end

    # analytics
    ratio = compression_ratio(test_text, tokens)
    @test ratio > 0.0
    freqs = token_frequencies(tokens)
    @test sum(values(freqs)) == length(tokens)
    history = vocab_size_history(corpus, 20)
    @test length(history) >= 2

    # vocabulary with special tokens
    v = get_vocabulary(vocab)
    extended = add_special_tokens(v, ["<unk>", "<pad>"])
    @test length(extended) == length(v) + 2

    # batch encoding
    batch = encode_batch(["low", "lower"], merges)
    @test length(batch) == 2
end

@testset "build_vocab_index" begin
    vocab = Set(["lo", "w", "</w>", "er"])
    index = build_vocab_index(vocab, ["<unk>", "<pad>"])
    # special tokens get IDs 1 and 2
    @test index["<unk>"] == 1
    @test index["<pad>"] == 2
    # vocab tokens sorted alphabetically after specials
    @test index["</w>"] == 3
    @test index["er"] == 4
    @test index["lo"] == 5
    @test index["w"] == 6
    @test length(index) == 6
    # without special tokens
    index2 = build_vocab_index(vocab)
    @test length(index2) == 4
    @test index2["</w>"] == 1
end

@testset "tokens_to_ids" begin
    index = Dict("lo" => 1, "w" => 2, "</w>" => 3)
    ids = tokens_to_ids(["lo", "w", "</w>", "unknown"], index)
    @test ids == [1, 2, 3, 0]
    # custom unk_id
    ids2 = tokens_to_ids(["lo", "missing"], index, unk_id=99)
    @test ids2 == [1, 99]
    # empty input
    @test tokens_to_ids(String[], index) == Int[]
end

@testset "ids_to_tokens" begin
    index = Dict("lo" => 1, "w" => 2, "</w>" => 3)
    tokens = ids_to_tokens([1, 2, 3], index)
    @test tokens == ["lo", "w", "</w>"]
    # unknown ID maps to "<unk>"
    tokens2 = ids_to_tokens([1, 999], index)
    @test tokens2 == ["lo", "<unk>"]
    # empty input
    @test ids_to_tokens(Int[], index) == String[]
end

@testset "save_vocab_index and load_vocab_index" begin
    index = Dict("lo" => 1, "w" => 2, "</w>" => 3, "er" => 4)
    tmpfile = tempname()
    try
        save_vocab_index(index, tmpfile)
        loaded = load_vocab_index(tmpfile)
        @test loaded == index
        @test length(loaded) == 4
    finally
        isfile(tmpfile) && rm(tmpfile)
    end
    @test_throws ErrorException load_vocab_index("nonexistent_vocab_index.txt")
end

@testset "token-to-ID round-trip" begin
    corpus = "low low low lower lower lowest"
    vocab, merges = train_bpe(corpus, 10)
    v = get_vocabulary(vocab)
    extended = add_special_tokens(v, ["<unk>", "<pad>"])
    index = build_vocab_index(extended, ["<unk>", "<pad>"])

    tokens = encode_text("low lower", merges)
    ids = tokens_to_ids(tokens, index)
    recovered = ids_to_tokens(ids, index)
    @test recovered == tokens

    # save/load round-trip
    tmpfile = tempname()
    try
        save_vocab_index(index, tmpfile)
        loaded_index = load_vocab_index(tmpfile)
        @test tokens_to_ids(tokens, loaded_index) == ids
    finally
        isfile(tmpfile) && rm(tmpfile)
    end
end

@testset "pad_sequence" begin
    @test pad_sequence([1, 2, 3], 5) == [1, 2, 3, 0, 0]
    @test pad_sequence([1, 2, 3], 5, pad_id=99) == [1, 2, 3, 99, 99]
    # already at max_len
    @test pad_sequence([1, 2, 3], 3) == [1, 2, 3]
    # longer than max_len — returned unchanged
    @test pad_sequence([1, 2, 3, 4], 2) == [1, 2, 3, 4]
    # empty input
    @test pad_sequence(Int[], 3) == [0, 0, 0]
end

@testset "truncate_sequence" begin
    @test truncate_sequence([1, 2, 3, 4, 5], 3) == [1, 2, 3]
    # already at max_len
    @test truncate_sequence([1, 2, 3], 3) == [1, 2, 3]
    # shorter than max_len — returned unchanged
    @test truncate_sequence([1, 2], 5) == [1, 2]
    # empty input
    @test truncate_sequence(Int[], 3) == Int[]
end

@testset "prepare_batch" begin
    batch = [[1, 2], [3, 4, 5, 6, 7], [8]]
    result = prepare_batch(batch, 4)
    @test length(result) == 3
    @test result[1] == [1, 2, 0, 0]
    @test result[2] == [3, 4, 5, 6]
    @test result[3] == [8, 0, 0, 0]
    # custom pad_id
    result2 = prepare_batch([[1], [2, 3]], 3, pad_id=-1)
    @test result2[1] == [1, -1, -1]
    @test result2[2] == [2, 3, -1]
    # empty batch
    @test prepare_batch(Vector{Int}[], 5) == Vector{Int}[]
end

@testset "pretokenize" begin
    chunks = pretokenize("Hello world")
    @test chunks == ["Hello", " world"]
    # punctuation gets its own chunk
    chunks2 = pretokenize("Hello, world!")
    @test "Hello" in chunks2
    @test "," in chunks2
    @test " world" in chunks2
    @test "!" in chunks2
    # contractions
    chunks3 = pretokenize("I'm don't")
    @test "'m" in chunks3
    @test "'t" in chunks3
    # numbers
    chunks4 = pretokenize("test 123 words")
    @test " 123" in chunks4
    # empty string
    @test pretokenize("") == String[]
end

@testset "count_frequencies_pretokenized" begin
    freqs = count_frequencies_pretokenized("hello world hello")
    @test freqs["hello"] == 1
    @test freqs[" hello"] == 1
    @test freqs[" world"] == 1
    @test length(freqs) == 3
    # single word
    freqs2 = count_frequencies_pretokenized("test")
    @test freqs2["test"] == 1
end

@testset "tokenize" begin
    corpus = "low low low lower lower lowest"
    _, merges = train_bpe(corpus, 10)
    tokens = tokenize("low lower", merges)
    @test length(tokens) > 0
    # result should be decodable
    decoded = decode_tokens(tokens)
    @test occursin("low", decoded)
    @test occursin("lower", decoded)
    # empty text
    @test tokenize("", merges) == String[]
    # handles mixed case via preprocessing
    tokens2 = tokenize("LOW Lower", merges)
    @test length(tokens2) > 0
end

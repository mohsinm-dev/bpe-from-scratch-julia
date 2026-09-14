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
    @test_throws TokenizerError load_corpus("nonexistent_file.txt")
end

@testset "normalize_unicode" begin
    # NFC normalization combines decomposed characters
    decomposed = "e\u0301"  # e + combining acute
    @test normalize_unicode(decomposed) == "é"
    # already normalized text is unchanged
    @test normalize_unicode("hello") == "hello"
    # NFD decomposes characters
    @test length(normalize_unicode("é", form=:NFD)) == 2
end

@testset "is_valid_utf8" begin
    @test is_valid_utf8("hello") == true
    @test is_valid_utf8("日本語") == true
    @test is_valid_utf8(Vector{UInt8}("hello")) == true
    # invalid UTF-8 byte sequence
    @test is_valid_utf8(UInt8[0xff, 0xfe]) == false
end

@testset "word_to_graphemes" begin
    @test word_to_graphemes("low") == ["l", "o", "w", "</w>"]
    # accented character stays as one unit
    @test word_to_graphemes("café") == ["c", "a", "f", "é", "</w>"]
    # single char
    @test word_to_graphemes("a") == ["a", "</w>"]
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
    @test_throws TokenizerError load_merges("nonexistent_merges.txt")
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
    @test_throws TokenizerError load_vocab_index("nonexistent_vocab_index.txt")
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
    # left padding
    @test pad_sequence([1, 2, 3], 5, side=:left) == [0, 0, 1, 2, 3]
    @test pad_sequence([1, 2], 4, pad_id=99, side=:left) == [99, 99, 1, 2]
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

@testset "pretokenize patterns" begin
    # LLaMA pattern
    chunks_llama = pretokenize("Hello, world! 123", pattern=LLAMA_PATTERN)
    @test length(chunks_llama) > 0
    # CLIP pattern splits more aggressively
    chunks_clip = pretokenize("Hello, world!", pattern=CLIP_PATTERN)
    @test "Hello" in chunks_clip
    @test length(chunks_clip) > 0
    # custom pattern
    chunks_custom = pretokenize("hello-world", pattern=r"[a-z]+")
    @test chunks_custom == ["hello", "world"]
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

@testset "BPETokenizer lifecycle" begin
    corpus = "low low low lower lower lowest"
    t = train_tokenizer(corpus, 10)

    # struct fields are populated
    @test length(t.merges) > 0
    @test length(t.vocab) > 0
    @test length(t.vocab_index) == length(t.vocab)
    @test length(t.id_to_token) == length(t.vocab_index)
    @test "<unk>" in t.special_tokens
    @test "<pad>" in t.special_tokens

    # encode produces integer IDs
    ids = encode(t, "low lower")
    @test ids isa Vector{Int}
    @test length(ids) > 0
    @test all(id -> id > 0, ids)

    # decode round-trip
    decoded = decode(t, ids)
    @test decoded == "low lower"

    # unknown tokens get unk_id
    ids_unk = encode(t, "xyz")
    unk_id = t.vocab_index["<unk>"]
    @test any(id -> id == unk_id, ids_unk)

    # save and reload
    dir = mktempdir()
    try
        save_tokenizer(t, dir)
        t2 = load_tokenizer(dir)
        @test t2.merges == t.merges
        @test t2.vocab_index == t.vocab_index
        @test t2.special_tokens == t.special_tokens
        # re-encode produces same IDs
        @test encode(t2, "low lower") == ids
        @test decode(t2, ids) == decoded
    finally
        rm(dir, recursive=true)
    end

    # load_tokenizer error on missing dir
    @test_throws TokenizerError load_tokenizer("nonexistent_tokenizer_dir")

    # custom special tokens
    t3 = train_tokenizer(corpus, 5, special_tokens=["<bos>", "<eos>"])
    @test "<bos>" in t3.special_tokens
    @test t3.vocab_index["<bos>"] == 1
    @test t3.vocab_index["<eos>"] == 2
end

@testset "most_common_tokens" begin
    tokens = ["a", "b", "a", "c", "a", "b", "d"]
    top2 = most_common_tokens(tokens, 2)
    @test length(top2) == 2
    @test top2[1] == ("a", 3)
    @test top2[2] == ("b", 2)
    # request more than available
    all_tokens = most_common_tokens(tokens, 100)
    @test length(all_tokens) == 4
    # empty input
    @test most_common_tokens(String[], 5) == Tuple{String,Int}[]
end

@testset "average_token_length" begin
    @test average_token_length(Set(["ab", "cdef", "g"])) ≈ 7 / 3
    @test average_token_length(Set(["hello"])) ≈ 5.0
    @test average_token_length(Set{String}()) == 0.0
end

@testset "coverage" begin
    corpus = "low low low lower lower lowest"
    _, merges = train_bpe(corpus, 10)
    # all trained words should be fully covered
    @test coverage("low lower lowest", merges) ≈ 1.0
    # unknown word drops coverage
    cov = coverage("low xyz", merges)
    @test cov < 1.0
    @test cov > 0.0
    # empty text
    @test coverage("", merges) == 0.0
end

@testset "text_to_bytes" begin
    @test text_to_bytes("Low") == ["4c", "6f", "77"]
    @test text_to_bytes("a") == ["61"]
    @test text_to_bytes("") == String[]
    # multibyte UTF-8
    bytes = text_to_bytes("ñ")
    @test length(bytes) == 2
end

@testset "bytes_to_text" begin
    @test bytes_to_text(["4c", "6f", "77"]) == "Low"
    @test bytes_to_text(["61"]) == "a"
    # merged tokens
    @test bytes_to_text(["4c6f", "77"]) == "Low"
    @test bytes_to_text(["4c6f77"]) == "Low"
    # empty
    @test bytes_to_text(String[]) == ""
    # round-trip
    text = "Hello world"
    @test bytes_to_text(text_to_bytes(text)) == text
end

@testset "train_byte_bpe" begin
    corpus = "low low low lower lower lowest"
    _, merges = train_byte_bpe(corpus, 5)
    @test length(merges) == 5
    @test merges[1] isa Tuple{String,String}
    # all merge components should be hex strings
    for (a, b) in merges
        @test all(c -> c in "0123456789abcdef", a)
        @test all(c -> c in "0123456789abcdef", b)
    end
    # early stopping when no pairs remain
    _, merges_max = train_byte_bpe("a", 100)
    @test length(merges_max) == 0
end

@testset "nbest_encode" begin
    corpus = "low low low lower lower lowest"
    _, merges = train_bpe(corpus, 10)
    results = nbest_encode("lower", merges, 3)
    @test length(results) >= 1
    @test length(results) <= 3
    # each result should be a valid tokenization
    for tokens in results
        @test length(tokens) > 0
    end
end

@testset "sample_segmentation" begin
    corpus = "low low low lower lower lowest"
    _, merges = train_bpe(corpus, 10)
    # temperature 0 should be deterministic
    t1 = sample_segmentation("low", merges, temperature=0.0)
    t2 = sample_segmentation("low", merges, temperature=0.0)
    @test t1 == t2
    @test t1 == encode_word("low", merges)
    # high temperature should produce tokens
    t3 = sample_segmentation("low", merges, temperature=5.0)
    @test length(t3) > 0
end

@testset "export/import huggingface merges" begin
    merges = [("l", "o"), ("lo", "w"), ("low", "</w>")]
    tmpfile = tempname()
    try
        export_huggingface_merges(merges, tmpfile)
        loaded = import_huggingface_merges(tmpfile)
        @test loaded == merges
        # file should have version header
        lines = readlines(tmpfile)
        @test startswith(lines[1], "#version")
    finally
        isfile(tmpfile) && rm(tmpfile)
    end
end

@testset "export_sentencepiece_vocab" begin
    index = Dict("lo" => 1, "w" => 2, "</w>" => 3)
    tmpfile = tempname()
    try
        export_sentencepiece_vocab(index, tmpfile)
        lines = readlines(tmpfile)
        @test length(lines) == 3
        @test occursin("\t", lines[1])
    finally
        isfile(tmpfile) && rm(tmpfile)
    end
end

@testset "parallel_count_pairs" begin
    ws = Dict(["l", "o", "w", "</w>"] => 3)
    pc = parallel_count_pairs(ws)
    @test pc[("l", "o")] == 3
    @test pc[("o", "w")] == 3
    # should match single-threaded results
    @test pc == count_pairs(ws)
end

@testset "train_bpe_parallel" begin
    corpus = "low low low lower lower lowest"
    _, merges_par = train_bpe_parallel(corpus, 5)
    _, merges_seq = train_bpe(corpus, 5)
    # parallel and sequential should produce same merges
    @test merges_par == merges_seq
end

@testset "TokenizerError" begin
    @test_throws TokenizerError train_bpe("", 10)
    @test_throws TokenizerError train_bpe("   ", 10)
    @test_throws TokenizerError train_bpe("hello", -1)
    # valid input should not throw
    _, merges = train_bpe("hello hello", 1)
    @test length(merges) >= 0
end

@testset "compare_tokenizers" begin
    corpus = "low low low lower lower lowest"
    _, merges = train_bpe(corpus, 10)
    tokenizers = Dict{String,Function}(
        "bpe" => text -> encode_text(text, merges),
        "char" => text -> string.(collect(text))
    )
    results = compare_tokenizers("low", tokenizers)
    @test haskey(results, "bpe")
    @test haskey(results, "char")
    @test length(results["char"]) == 3  # l, o, w
end

@testset "compare_compression" begin
    corpus = "low low low lower lower lowest"
    _, merges = train_bpe(corpus, 10)
    tokenizers = Dict{String,Function}(
        "bpe" => text -> encode_text(text, merges),
        "char" => text -> string.(collect(text))
    )
    ratios = compare_compression("low lower lowest", tokenizers)
    # BPE should compress better (higher ratio) than character-level
    @test ratios["bpe"] > ratios["char"]
end

@testset "viterbi_segment" begin
    scores = Dict("l" => -1.0, "o" => -1.0, "w" => -1.0, "lo" => -0.5, "low" => -0.3)
    tokens = viterbi_segment("low", scores)
    # should prefer "low" as single token (highest score)
    @test tokens == ["low"]
    # with only single chars
    scores2 = Dict("a" => -1.0, "b" => -1.0, "c" => -1.0)
    @test viterbi_segment("abc", scores2) == ["a", "b", "c"]
    @test viterbi_segment("", scores) == String[]
end

@testset "train_unigram" begin
    corpus = "low low low lower lower lowest"
    scores = train_unigram(corpus, 15)
    @test length(scores) <= 15
    # all single chars should be in vocab
    @test haskey(scores, "l")
    @test haskey(scores, "o")
    @test haskey(scores, "w")
    # scores should be negative log-probs
    @test all(v -> v <= 0.0, values(scores))
end

@testset "TokenizerConfig save/load" begin
    config = TokenizerConfig(num_merges=50, min_frequency=2,
        special_tokens=["<unk>", "<pad>", "<bos>"], lowercase=true, verbose=false)
    tmpfile = tempname()
    try
        save_config(config, tmpfile)
        loaded = load_config(tmpfile)
        @test loaded.num_merges == 50
        @test loaded.min_frequency == 2
        @test loaded.special_tokens == ["<unk>", "<pad>", "<bos>"]
        @test loaded.lowercase == true
        @test loaded.verbose == false
    finally
        isfile(tmpfile) && rm(tmpfile)
    end
    @test_throws ErrorException load_config("nonexistent.json")
end

@testset "train_from_config" begin
    corpus = "low low low lower lower lowest"
    config = TokenizerConfig(num_merges=5)
    t = train_from_config(corpus, config)
    @test length(t.merges) > 0
    @test "<unk>" in t.special_tokens
end

@testset "train_bpe_with_history" begin
    corpus = "low low low lower lower lowest"
    _, merges, history = train_bpe_with_history(corpus, 5)
    @test length(history) == length(merges)
    @test history[1].step == 1
    @test history[1].frequency > 0
    @test history[1].new_token == history[1].pair[1] * history[1].pair[2]
    @test history[1].vocab_size > 0
end

@testset "format_merge_history" begin
    corpus = "low low low lower lower lowest"
    _, _, history = train_bpe_with_history(corpus, 3)
    table = format_merge_history(history)
    @test occursin("Step", table)
    @test occursin("Freq", table)
    @test length(split(table, "\n")) == length(history) + 2  # header + separator + rows
end

@testset "token_length_distribution" begin
    vocab = Set(["a", "ab", "abc", "de", "f"])
    dist = token_length_distribution(vocab)
    @test dist[1] == 2  # "a", "f"
    @test dist[2] == 2  # "ab", "de"
    @test dist[3] == 1  # "abc"
    @test token_length_distribution(Set{String}()) == Dict{Int,Int}()
end

@testset "subword_fertility" begin
    corpus = "low low low lower lower lowest"
    _, merges = train_bpe(corpus, 10)
    fert = subword_fertility("low lower lowest", merges)
    @test fert > 0.0
    # trained words should have low fertility
    @test fert < 5.0
    @test subword_fertility("", merges) == 0.0
end

@testset "vocab_overlap" begin
    v1 = Set(["a", "b", "c"])
    v2 = Set(["b", "c", "d"])
    result = vocab_overlap(v1, v2)
    @test result.shared == Set(["b", "c"])
    @test result.only1 == Set(["a"])
    @test result.only2 == Set(["d"])
    @test result.jaccard ≈ 2 / 4
end

@testset "train_wordpiece" begin
    corpus = "low low low lower lower lowest"
    vocab = train_wordpiece(corpus, 30)
    @test "l" in vocab
    @test length(vocab) > 0
    @test length(vocab) <= 30
end

@testset "wordpiece_tokenize" begin
    corpus = "low low low lower lower lowest"
    vocab = train_wordpiece(corpus, 30)
    tokens = wordpiece_tokenize("low", vocab)
    @test length(tokens) >= 1
    # first token should not have ## prefix
    @test !startswith(tokens[1], "##")
    # unknown word should return [UNK]
    tokens_unk = wordpiece_tokenize("zzzzz", vocab)
    @test tokens_unk == ["[UNK]"]
    # very long word
    tokens_long = wordpiece_tokenize("a" ^ 200, vocab, max_word_len=100)
    @test tokens_long == ["[UNK]"]
end

@testset "count_word_frequencies_streaming" begin
    path = joinpath(@__DIR__, "..", "data", "sample_corpus.txt")
    freqs = count_word_frequencies_streaming(path)
    @test freqs["low"] > 0
    @test freqs["the"] > 0
    @test_throws ErrorException count_word_frequencies_streaming("nonexistent.txt")
end

@testset "train_bpe_streaming" begin
    path = joinpath(@__DIR__, "..", "data", "sample_corpus.txt")
    _, merges = train_bpe_streaming(path, 10)
    @test length(merges) > 0
    @test length(merges) <= 10
    # results should match in-memory training on same data
    corpus = load_corpus(path)
    _, merges_mem = train_bpe(corpus, 10)
    # same first merge since data is identical
    @test merges[1] == merges_mem[1]
end

@testset "train_bpe_protected" begin
    corpus = "low low low lower lower lowest"
    _, merges_normal = train_bpe(corpus, 5)
    # protect the first merge pair — it should never appear in protected merges
    first_pair = merges_normal[1]
    _, merges_protected = train_bpe_protected(corpus, 5, never_merge=Set([first_pair]))
    @test first_pair in merges_normal
    @test !(first_pair in merges_protected)
    # should still produce merges
    @test length(merges_protected) > 0
end

@testset "encode_with_protected_tokens" begin
    corpus = "low low low lower lower lowest"
    _, merges = train_bpe(corpus, 10)
    # without protection, encode normally
    @test encode_with_protected_tokens("low lower", merges) == encode_text("low lower", merges)
    # with protection, protected string appears as single token
    tokens = encode_with_protected_tokens("low [MASK] lower", merges, protected=["[MASK]"])
    @test "[MASK]" in tokens
end

@testset "validate_merges" begin
    # valid merges produce no warnings
    @test validate_merges([("a", "b"), ("ab", "c")]) == String[]
    # duplicate merges
    warns = validate_merges([("a", "b"), ("a", "b")])
    @test length(warns) == 1
    @test occursin("duplicate", warns[1])
    # empty component
    warns2 = validate_merges([("", "b")])
    @test length(warns2) == 1
    @test occursin("empty", warns2[1])
end

@testset "validate_vocab_index" begin
    # valid index
    @test validate_vocab_index(Dict("a" => 1, "b" => 2, "c" => 3)) == String[]
    # empty index
    warns = validate_vocab_index(Dict{String,Int}())
    @test length(warns) == 1
    # gaps in IDs
    warns2 = validate_vocab_index(Dict("a" => 1, "b" => 5))
    @test any(w -> occursin("gaps", w), warns2)
end

@testset "validate_tokenizer" begin
    corpus = "low low low lower lower lowest"
    t = train_tokenizer(corpus, 10)
    # a well-trained tokenizer should have no warnings
    @test validate_tokenizer(t) == String[]
end

@testset "integration: WordPiece full pipeline" begin
    corpus = "the cat sat on the mat the dog sat on the log running runs runner"
    vocab = train_wordpiece(corpus, 40)
    @test length(vocab) > 0
    # tokenize all words from corpus
    for word in split(corpus)
        tokens = wordpiece_tokenize(String(word), vocab)
        @test length(tokens) >= 1
        # first token should not have ## prefix
        @test !startswith(tokens[1], "##")
    end
end

@testset "integration: Unigram full pipeline" begin
    corpus = "low low low lower lower lowest high higher highest"
    scores = train_unigram(corpus, 20)
    @test length(scores) > 0
    # segment all words
    for word in split(corpus)
        tokens = viterbi_segment(String(word), scores)
        @test length(tokens) >= 1
        @test join(tokens) == word
    end
end

@testset "integration: cross-tokenizer comparison" begin
    corpus = "low low low lower lower lowest running runner runs"
    _, bpe_merges = train_bpe(corpus, 15)
    wp_vocab = train_wordpiece(corpus, 30)
    uni_scores = train_unigram(corpus, 20)

    text = "low lower"
    bpe_tokens = encode_text(text, bpe_merges)
    @test length(bpe_tokens) > 0

    for word in split(text)
        wp_tokens = wordpiece_tokenize(String(word), wp_vocab)
        @test length(wp_tokens) >= 1
        uni_tokens = viterbi_segment(String(word), uni_scores)
        @test length(uni_tokens) >= 1
    end
end

@testset "stress: large vocabulary training" begin
    words = ["low", "lower", "lowest", "high", "higher", "highest",
             "run", "running", "runner", "the", "cat", "sat", "on", "mat"]
    corpus = join(rand(words, 500), " ")
    _, merges = train_bpe(corpus, 100)
    @test length(merges) > 0
    # encode should still work
    tokens = encode_text("low lower highest running", merges)
    @test length(tokens) > 0
    decoded = decode_tokens(tokens)
    @test decoded == "low lower highest running"
end

@testset "encode_byte_level" begin
    corpus = "low low low lower lower lowest"
    _, merges = train_byte_bpe(corpus, 5)
    tokens = encode_byte_level("low", merges)
    @test length(tokens) >= 1
    # decoded bytes should reconstruct the word
    @test bytes_to_text(tokens) == "low"
    # encoding trained text produces merged tokens
    tokens2 = encode_byte_level("low lower", merges)
    @test bytes_to_text(tokens2) == "lowlower"
    # empty word handling
    @test encode_byte_level("", merges) == String[]
end

@testset "edge cases: empty and whitespace input" begin
    # empty corpus errors
    @test_throws TokenizerError train_bpe("", 5)
    @test_throws TokenizerError train_bpe("   \t\n  ", 5)

    # empty text encoding returns empty
    merges = [("l", "o")]
    @test encode_text("", merges) == String[]
    @test encode_batch(String[], merges) == Vector{String}[]

    # decode empty tokens
    @test decode_tokens(String[]) == ""

    # single character word
    @test encode_word("a", Tuple{String,String}[]) == ["a", "</w>"]
    @test word_to_symbols("a") == ["a", "</w>"]
    @test word_to_graphemes("a") == ["a", "</w>"]

    # whitespace-only text
    @test pretokenize("") == String[]
    @test tokenize("", Tuple{String,String}[]) == String[]

    # zero merges is valid
    vocab, merges0 = train_bpe("hello hello", 0)
    @test length(merges0) == 0
    @test length(vocab) > 0

    # single-word corpus
    _, m = train_bpe("hello", 5)
    @test length(m) > 0
end

@testset "error recovery: corrupted and missing files" begin
    # missing files raise TokenizerError with descriptive messages
    @test_throws TokenizerError load_merges("does_not_exist.tsv")
    @test_throws TokenizerError load_vocab_index("does_not_exist.tsv")
    @test_throws TokenizerError load_corpus("does_not_exist.txt")
    @test_throws ErrorException load_config("does_not_exist.json")
    @test_throws TokenizerError load_tokenizer("does_not_exist_dir")
    @test_throws ErrorException count_word_frequencies_streaming("does_not_exist.txt")

    # verify error messages contain the filepath
    try
        load_corpus("missing_file.txt")
    catch e
        @test e isa TokenizerError
        @test occursin("missing_file.txt", e.msg)
    end

    # malformed merges file (missing tab separator) loads with zero entries
    tmpfile = tempname()
    try
        open(tmpfile, "w") do io
            println(io, "no_tab_here")
            println(io, "also broken")
        end
        loaded = load_merges(tmpfile)
        @test length(loaded) == 0
    finally
        isfile(tmpfile) && rm(tmpfile)
    end

    # malformed vocab index (non-integer IDs) should error on parse
    tmpfile2 = tempname()
    try
        open(tmpfile2, "w") do io
            println(io, "token\tnot_a_number")
        end
        @test_throws Exception load_vocab_index(tmpfile2)
    finally
        isfile(tmpfile2) && rm(tmpfile2)
    end

    # empty file loads as empty merges
    tmpfile3 = tempname()
    try
        open(tmpfile3, "w") do io end
        @test load_merges(tmpfile3) == Tuple{String,String}[]
    finally
        isfile(tmpfile3) && rm(tmpfile3)
    end
end

@testset "unicode edge cases" begin
    # combining characters
    decomposed = "e\u0301"  # e + combining acute = é
    @test normalize_unicode(decomposed) == "é"
    @test is_valid_utf8(decomposed)

    # zero-width joiner (emoji sequences)
    zwj_text = "a\u200Bb"  # zero-width space
    @test is_valid_utf8(zwj_text)
    processed = preprocess_text(zwj_text)
    @test length(processed) > 0

    # CJK characters
    cjk = "你好世界"
    @test is_valid_utf8(cjk)
    syms = word_to_symbols(cjk)
    @test length(syms) == 5  # 4 chars + </w>
    graphemes_cjk = word_to_graphemes(cjk)
    @test length(graphemes_cjk) == 5

    # emoji
    emoji = "🎉🚀"
    @test is_valid_utf8(emoji)
    bytes = text_to_bytes(emoji)
    @test length(bytes) > 0
    @test bytes_to_text(bytes) == emoji

    # mixed script training
    mixed = "hello 你好 hello 你好 world"
    _, merges = train_bpe(mixed, 5)
    @test length(merges) > 0
    tokens = encode_text("hello 你好", merges)
    @test decode_tokens(tokens) == "hello 你好"

    # RTL text (Arabic)
    rtl = "مرحبا مرحبا"
    _, merges_rtl = train_bpe(rtl, 3)
    @test length(merges_rtl) > 0

    # single grapheme cluster with multiple codepoints
    flag = "🇺🇸"
    graphemes_flag = word_to_graphemes(flag)
    @test graphemes_flag[end] == "</w>"
    @test length(graphemes_flag) >= 2
end

@testset "round-trip consistency across algorithms" begin
    corpus = "the quick brown fox jumps over the lazy dog the fox"
    text = "the fox"

    # BPE round-trip
    _, bpe_merges = train_bpe(corpus, 10)
    bpe_tokens = encode_text(text, bpe_merges)
    @test decode_tokens(bpe_tokens) == text

    # byte-level round-trip
    _, byte_merges = train_byte_bpe(corpus, 5)
    byte_tokens = encode_byte_level(text, byte_merges)
    # byte level encodes words separately, no spaces
    @test bytes_to_text(byte_tokens) == replace(text, " " => "")

    # unigram covers all characters
    scores = train_unigram(corpus, 20)
    for word in split(text)
        segments = viterbi_segment(String(word), scores)
        @test join(segments) == word
    end

    # wordpiece covers trained words
    wp_vocab = train_wordpiece(corpus, 30)
    for word in split(corpus)
        tokens = wordpiece_tokenize(String(word), wp_vocab)
        @test tokens != ["[UNK]"]
    end
end

@testset "input validation at API boundaries" begin
    # train_tokenizer rejects negative num_merges
    @test_throws TokenizerError train_tokenizer("hello hello", -1)

    # train_wordpiece rejects non-positive vocab_size
    @test_throws TokenizerError train_wordpiece("hello hello", 0)
    @test_throws TokenizerError train_wordpiece("hello hello", -5)

    # train_unigram rejects non-positive vocab_size
    @test_throws TokenizerError train_unigram("hello hello", 0)
    @test_throws TokenizerError train_unigram("hello hello", -1)

    # train_bpe rejects negative num_merges (already tested but grouped here)
    @test_throws TokenizerError train_bpe("hello hello", -1)
end

@testset "encode_streaming" begin
    corpus_path = joinpath(@__DIR__, "..", "data", "sample_corpus.txt")
    corpus = load_corpus(corpus_path)
    _, merges = train_bpe(corpus, 10)

    # streaming encode should produce same tokens as in-memory encode
    streaming_tokens = encode_streaming(corpus_path, merges)
    @test length(streaming_tokens) > 0

    # callback is invoked for each non-empty line
    lines_seen = Int[]
    encode_streaming(corpus_path, merges, callback=(n, _) -> push!(lines_seen, n))
    @test length(lines_seen) > 0
    @test issorted(lines_seen)

    # missing file raises error
    @test_throws ErrorException encode_streaming("nonexistent.txt", merges)

    # empty file produces empty tokens
    tmpfile = tempname()
    try
        open(tmpfile, "w") do io end
        @test encode_streaming(tmpfile, merges) == String[]
    finally
        isfile(tmpfile) && rm(tmpfile)
    end

    # file with blank lines skips them
    tmpfile2 = tempname()
    try
        open(tmpfile2, "w") do io
            println(io, "low lower")
            println(io, "")
            println(io, "lowest")
        end
        tokens = encode_streaming(tmpfile2, merges)
        @test length(tokens) > 0
        decoded = decode_tokens(tokens)
        @test occursin("low", decoded)
    finally
        isfile(tmpfile2) && rm(tmpfile2)
    end
end

@testset "token_entropy" begin
    # uniform distribution has maximum entropy
    uniform = ["a", "b", "c", "d"]
    @test token_entropy(uniform) ≈ 2.0  # log2(4) = 2

    # single token has zero entropy
    @test token_entropy(["a", "a", "a"]) ≈ 0.0

    # empty input
    @test token_entropy(String[]) == 0.0

    # more varied = higher entropy
    low_var = ["a", "a", "a", "b"]
    high_var = ["a", "b", "c", "d"]
    @test token_entropy(low_var) < token_entropy(high_var)
end

@testset "vocabulary_coverage_report" begin
    corpus = "low low low lower lower lowest"
    _, merges = train_bpe(corpus, 10)

    report = vocabulary_coverage_report("low lower lowest", merges)
    @test report.total == 3
    @test report.covered == 3
    @test report.coverage ≈ 1.0
    @test isempty(report.uncovered)

    # with unknown words
    report2 = vocabulary_coverage_report("low xyz", merges)
    @test report2.total == 2
    @test report2.covered < report2.total
    @test "xyz" in report2.uncovered

    # empty text
    report3 = vocabulary_coverage_report("", merges)
    @test report3.total == 0
    @test report3.coverage == 0.0
end

@testset "oov_rate" begin
    index = Dict("lo" => 1, "w" => 2, "</w>" => 3)

    # all tokens known
    @test oov_rate(["lo", "w", "</w>"], index) ≈ 0.0

    # some unknown tokens
    @test oov_rate(["lo", "unknown", "</w>"], index) ≈ 1/3

    # all unknown
    @test oov_rate(["x", "y", "z"], index) ≈ 1.0

    # empty input
    @test oov_rate(String[], index) == 0.0
end

@testset "CachedEncoder" begin
    corpus = "low low low lower lower lowest"
    _, merges = train_bpe(corpus, 10)

    enc = CachedEncoder(merges)

    # first call is a miss
    result1 = cached_encode_word(enc, "low")
    @test result1 == encode_word("low", merges)
    stats = cache_stats(enc)
    @test stats.misses == 1
    @test stats.hits == 0

    # second call is a hit
    result2 = cached_encode_word(enc, "low")
    @test result2 == result1
    stats2 = cache_stats(enc)
    @test stats2.hits == 1

    # clear resets
    clear_cache!(enc)
    stats3 = cache_stats(enc)
    @test stats3.hits == 0
    @test stats3.misses == 0
    @test stats3.size == 0
end

@testset "corpus_from_directory" begin
    dir = mktempdir()
    try
        open(joinpath(dir, "a.txt"), "w") do io
            print(io, "hello world")
        end
        open(joinpath(dir, "b.txt"), "w") do io
            print(io, "foo bar")
        end
        open(joinpath(dir, "c.md"), "w") do io
            print(io, "ignored")
        end
        corpus = corpus_from_directory(dir)
        @test occursin("hello world", corpus)
        @test occursin("foo bar", corpus)
        @test !occursin("ignored", corpus)
        # custom extension
        corpus_md = corpus_from_directory(dir, extension=".md")
        @test occursin("ignored", corpus_md)
    finally
        rm(dir, recursive=true)
    end
    # missing directory
    @test_throws TokenizerError corpus_from_directory("nonexistent_dir")
end

@testset "decode_to_words" begin
    tokens = ["low", "</w>", "er", "</w>"]
    @test decode_to_words(tokens) == ["low", "er"]
    # trailing tokens without </w>
    @test decode_to_words(["h", "i"]) == ["hi"]
    # empty
    @test decode_to_words(String[]) == String[]
    # single word
    @test decode_to_words(["hello", "</w>"]) == ["hello"]
end

@testset "encode_streaming_batch" begin
    corpus_path = joinpath(@__DIR__, "..", "data", "sample_corpus.txt")
    corpus = load_corpus(corpus_path)
    _, merges = train_bpe(corpus, 10)

    batches = encode_streaming_batch(corpus_path, merges, batch_size=2)
    @test length(batches) >= 1
    @test all(b -> length(b) > 0, batches)

    # missing file
    @test_throws TokenizerError encode_streaming_batch("nonexistent.txt", merges)
end

@testset "save_tokenizer_json and load_tokenizer_json" begin
    corpus = "low low low lower lower lowest"
    t = train_tokenizer(corpus, 10)
    tmpfile = tempname() * ".json"
    try
        save_tokenizer_json(t, tmpfile)
        t2 = load_tokenizer_json(tmpfile)
        @test t2.merges == t.merges
        @test t2.vocab_index == t.vocab_index
        @test t2.special_tokens == t.special_tokens
        # encode produces same results
        @test encode(t2, "low lower") == encode(t, "low lower")
    finally
        isfile(tmpfile) && rm(tmpfile)
    end
    # missing file
    @test_throws TokenizerError load_tokenizer_json("nonexistent.json")
end

@testset "tokenizer_summary" begin
    corpus = "low low low lower lower lowest"
    t = train_tokenizer(corpus, 10)
    summary = tokenizer_summary(t)
    @test occursin("Vocabulary size:", summary)
    @test occursin("Merge rules:", summary)
    @test occursin("Special tokens:", summary)
    @test occursin("Avg token length:", summary)
end

@testset "token_length_histogram" begin
    vocab = Set(["a", "ab", "abc", "de", "f"])
    hist = token_length_histogram(vocab)
    @test occursin("#", hist)
    @test occursin("1", hist)
    @test occursin("2", hist)
    # empty vocab
    @test token_length_histogram(Set{String}()) == ""
end

@testset "prune_vocabulary" begin
    index = Dict("lo" => 1, "w" => 2, "</w>" => 3, "er" => 4, "low" => 5)
    tokens = ["lo", "w", "</w>", "lo", "w", "</w>", "low", "</w>"]
    pruned = prune_vocabulary(index, tokens, 2)
    # "lo", "w", "</w>" appear >= 2 times, "er" appears 0 but is 2 chars so pruned
    @test haskey(pruned, "lo")
    @test haskey(pruned, "w")
    @test haskey(pruned, "</w>")
    @test !haskey(pruned, "er")
    # single-char tokens always kept
    @test haskey(pruned, "w")
end

@testset "corpus_statistics" begin
    stats = corpus_statistics("low low low lower lower lowest")
    @test stats.word_count == 6
    @test stats.char_count == length("low low low lower lower lowest")
    @test stats.unique_words == 3
    @test stats.avg_word_length > 0.0

    # empty corpus
    stats_empty = corpus_statistics("")
    @test stats_empty.word_count == 0
    @test stats_empty.avg_word_length == 0.0
end

@testset "attention_mask" begin
    @test attention_mask([1, 2, 3, 0, 0]) == [1, 1, 1, 0, 0]
    @test attention_mask([0, 0, 0]) == [0, 0, 0]
    @test attention_mask([5, 10, 15]) == [1, 1, 1]
    @test attention_mask(Int[]) == Int[]
    # custom pad_id
    @test attention_mask([1, 99, 99], pad_id=99) == [1, 0, 0]
end

@testset "performance: training completes in bounded time" begin
    # generate a reasonably sized corpus
    words = ["the", "quick", "brown", "fox", "jumps", "over", "lazy", "dog",
             "running", "runner", "highest", "lower", "lowest", "newer"]
    corpus = join(rand(words, 1000), " ")

    # training 200 merges should complete without hanging
    t = @elapsed begin
        _, merges = train_bpe(corpus, 200)
    end
    @test length(merges) > 0
    @test t < 30.0  # should finish well under 30 seconds

    # encoding 1000 words should be fast
    t2 = @elapsed begin
        tokens = encode_text(corpus, merges)
    end
    @test length(tokens) > 0
    @test t2 < 10.0
end

@testset "batch_decode" begin
    corpus = "low low low lower lower lowest"
    t = train_tokenizer(corpus, 10)
    ids1 = encode(t, "low")
    ids2 = encode(t, "lower")
    results = batch_decode(t, [ids1, ids2])
    @test results == ["low", "lower"]
    @test batch_decode(t, Vector{Int}[]) == String[]
end

@testset "token_pair_statistics" begin
    tokens = ["lo", "w", "</w>", "lo", "w", "er", "</w>"]
    stats = token_pair_statistics(tokens)
    @test stats[("lo", "w")] == 2
    @test stats[("w", "</w>")] == 1
    @test stats[("w", "er")] == 1
    # single token
    @test token_pair_statistics(["a"]) == Dict{Tuple{String,String},Int}()
    # empty
    @test token_pair_statistics(String[]) == Dict{Tuple{String,String},Int}()
end

@testset "filter_vocabulary" begin
    vocab = Set(["a", "ab", "abc", "abcd", "</w>"])
    @test filter_vocabulary(vocab, min_length=2) == Set(["ab", "abc", "abcd", "</w>"])
    @test filter_vocabulary(vocab, max_length=2) == Set(["a", "ab"])
    @test filter_vocabulary(vocab, min_length=2, max_length=3) == Set(["ab", "abc", "</w>"])
    @test filter_vocabulary(Set{String}()) == Set{String}()
end

@testset "encode_with_cache" begin
    corpus = "low low low lower lower lowest"
    _, merges = train_bpe(corpus, 10)
    enc = CachedEncoder(merges)
    tokens = encode_with_cache("low lower low", enc)
    @test length(tokens) > 0
    @test decode_tokens(tokens) == "low lower low"
    stats = cache_stats(enc)
    @test stats.hits >= 1  # "low" encoded twice
    @test encode_with_cache("", enc) == String[]
end

@testset "vocabulary_diff" begin
    v1 = Set(["a", "b", "c"])
    v2 = Set(["b", "c", "d", "e"])
    diff = vocabulary_diff(v1, v2)
    @test diff.added == Set(["d", "e"])
    @test diff.removed == Set(["a"])
    @test diff.common == Set(["b", "c"])
    # identical vocabs
    diff2 = vocabulary_diff(v1, v1)
    @test isempty(diff2.added)
    @test isempty(diff2.removed)
    @test diff2.common == v1
end

@testset "parallel_encode_batch" begin
    corpus = "low low low lower lower lowest"
    _, merges = train_bpe(corpus, 10)
    texts = ["low", "lower", "lowest"]
    par_results = parallel_encode_batch(texts, merges)
    seq_results = encode_batch(texts, merges)
    @test par_results == seq_results
    @test parallel_encode_batch(String[], merges) == Vector{String}[]
end

@testset "corpus_from_files" begin
    dir = mktempdir()
    try
        f1 = joinpath(dir, "a.txt")
        f2 = joinpath(dir, "b.txt")
        open(f1, "w") do io print(io, "hello world") end
        open(f2, "w") do io print(io, "foo bar") end
        corpus = corpus_from_files([f1, f2])
        @test occursin("hello world", corpus)
        @test occursin("foo bar", corpus)
        # missing file
        @test_throws TokenizerError corpus_from_files(["nonexistent.txt"])
        # empty list
        @test corpus_from_files(String[]) == ""
    finally
        rm(dir, recursive=true)
    end
end

@testset "special_token_ids" begin
    corpus = "low low low lower lower lowest"
    t = train_tokenizer(corpus, 10)
    ids = special_token_ids(t)
    @test ids["<unk>"] == t.vocab_index["<unk>"]
    @test ids["<pad>"] == t.vocab_index["<pad>"]
    @test length(ids) == 2
end

@testset "encode_with_offsets" begin
    merges = [("l", "o"), ("lo", "w")]
    offsets = encode_with_offsets("low", merges)
    @test offsets[1] == ("low", 1, 3)
    @test offsets[2] == ("</w>", 4, 4)
    # no merges
    offsets2 = encode_with_offsets("ab", Tuple{String,String}[])
    @test offsets2[1] == ("a", 1, 1)
    @test offsets2[2] == ("b", 2, 2)
    @test offsets2[3] == ("</w>", 3, 3)
end

@testset "validate_encoding" begin
    corpus = "low low low lower lower lowest"
    t = train_tokenizer(corpus, 10)
    @test validate_encoding(t, "low lower") == true
    @test validate_encoding(t, "low lower lowest") == true
end

@testset "vocab_growth_rate" begin
    history = [10, 11, 12, 14]
    rates = vocab_growth_rate(history)
    @test length(rates) == 3
    @test rates[1] ≈ 10.0
    @test rates[2] ≈ 100/11
    # single element
    @test vocab_growth_rate([10]) == Float64[]
    @test vocab_growth_rate(Int[]) == Float64[]
end

@testset "token_type_ids" begin
    @test token_type_ids(3, 2) == [0, 0, 0, 1, 1]
    @test token_type_ids(0, 3) == [1, 1, 1]
    @test token_type_ids(2, 0) == [0, 0]
    @test token_type_ids(0, 0) == Int[]
end

@testset "merge_frequency_histogram" begin
    corpus = "low low low lower lower lowest"
    _, _, history = train_bpe_with_history(corpus, 5)
    hist = merge_frequency_histogram(history)
    @test occursin("#", hist)
    @test length(split(hist, "\n")) == length(history)
    @test merge_frequency_histogram(MergeRecord[]) == ""
end

@testset "encode_sentences" begin
    corpus = "low low low lower lower lowest"
    _, merges = train_bpe(corpus, 10)
    tokens = encode_sentences("low lower. lowest low.", merges)
    @test "</s>" in tokens
    # custom separator
    tokens2 = encode_sentences("low. lower.", merges, sep_token="[SEP]")
    @test "[SEP]" in tokens2
    # single sentence (no separator)
    tokens3 = encode_sentences("low lower", merges)
    @test !("</s>" in tokens3)
end

@testset "normalize_token" begin
    @test normalize_token("low</w>") == "low"
    @test normalize_token("</w>") == ""
    @test normalize_token("##er") == "er"
    @test normalize_token("hello") == "hello"
    @test normalize_token("") == ""
end

@testset "incremental_train_bpe" begin
    corpus = "low low low lower lower lowest"
    vocab1, merges1 = train_bpe(corpus, 3)
    vocab2, merges2 = incremental_train_bpe(vocab1, merges1, 3)
    @test length(merges2) == length(merges1) + 3
    @test merges2[1:3] == merges1
    # encoding with extended merges should still decode correctly
    tokens = encode_text("low lower", merges2)
    @test decode_tokens(tokens) == "low lower"
end

@testset "tokenizer_equality" begin
    corpus = "low low low lower lower lowest"
    t1 = train_tokenizer(corpus, 10)
    # save and reload should be equal
    dir = mktempdir()
    try
        save_tokenizer(t1, dir)
        t2 = load_tokenizer(dir)
        @test tokenizer_equality(t1, t2)
    finally
        rm(dir, recursive=true)
    end
    # different training produces different tokenizer
    t3 = train_tokenizer(corpus, 5)
    @test !tokenizer_equality(t1, t3)
end

@testset "top_merges" begin
    corpus = "low low low lower lower lowest"
    _, _, history = train_bpe_with_history(corpus, 5)
    top = top_merges(history, 2)
    @test length(top) == 2
    @test top[1].frequency >= top[2].frequency
    # request more than available
    @test length(top_merges(history, 100)) == length(history)
    @test top_merges(MergeRecord[], 5) == MergeRecord[]
end

@testset "tokenizer_info" begin
    corpus = "low low low lower lower lowest"
    t = train_tokenizer(corpus, 10)
    info = tokenizer_info(t)
    @test occursin("Vocabulary size:", info)
    @test occursin("Merge rules:", info)
    @test occursin("Validation:", info)
    @test occursin("valid", info)
end

@testset "byte_pair_frequency_table" begin
    corpus = "low low low lower lower lowest"
    freqs = count_word_frequencies(corpus)
    ws = initialize_word_symbols(freqs)
    table = byte_pair_frequency_table(ws)
    @test occursin("Rank", table)
    @test occursin("Frequency", table)
    @test length(split(table, "\n")) >= 3  # header + separator + at least 1 row
    # empty word_symbols
    @test byte_pair_frequency_table(Dict{Vector{String},Int}()) == "Rank | Pair            | Frequency\n" * "-" ^ 45
end

@testset "estimated_vocab_size" begin
    corpus = "low low low lower lower lowest"
    est = estimated_vocab_size(corpus, 10)
    # should be close to actual
    _, merges = train_bpe(corpus, 10)
    actual_vocab = get_vocabulary(train_bpe(corpus, 10)[1])
    # estimate is an upper bound
    @test est >= length(actual_vocab) - 5  # allow some slack
    @test est > 0
end

@testset "corpus_statistics_streaming" begin
    path = joinpath(@__DIR__, "..", "data", "sample_corpus.txt")
    stats = corpus_statistics_streaming(path)
    @test stats.word_count > 0
    @test stats.char_count > 0
    @test stats.unique_words > 0
    @test stats.avg_word_length > 0.0
    # compare with in-memory version
    corpus = load_corpus(path)
    mem_stats = corpus_statistics(corpus)
    @test stats.word_count == mem_stats.word_count
    # missing file
    @test_throws TokenizerError corpus_statistics_streaming("nonexistent.txt")
end

@testset "merge_statistics" begin
    merges = [("l", "o"), ("lo", "w"), ("low", "</w>")]
    stats = merge_statistics(merges)
    @test stats.count == 3
    @test stats.avg_token_length > 0.0
    @test stats.max_token_length == 6  # "low</w>"
    @test stats.longest_token == "low</w>"
    # empty merges
    empty_stats = merge_statistics(Tuple{String,String}[])
    @test empty_stats.count == 0
    @test empty_stats.avg_token_length == 0.0
end

@testset "is_trained" begin
    corpus = "low low low lower lower lowest"
    t = train_tokenizer(corpus, 10)
    @test is_trained(t) == true
    # zero merges produces untrained tokenizer
    t0 = train_tokenizer(corpus, 0)
    @test is_trained(t0) == false
end

@testset "corpus_vocabulary" begin
    vocab = corpus_vocabulary("low low lower lowest higher")
    @test vocab == Set(["low", "lower", "lowest", "higher"])
    @test length(vocab) == 4
    @test corpus_vocabulary("") == Set{String}()
end

@testset "split_corpus" begin
    corpus = join(["word$i" for i in 1:100], " ")
    train, test = split_corpus(corpus)
    @test length(split(train)) == 80
    @test length(split(test)) == 20
    # deterministic
    train2, test2 = split_corpus(corpus)
    @test train == train2
    @test test == test2
    # custom ratio
    train3, test3 = split_corpus(corpus, ratio=0.5)
    @test length(split(train3)) == 50
end

@testset "char_coverage" begin
    vocab = Set(["l", "o", "w", "</w>", "lo"])
    @test char_coverage("low", vocab) ≈ 1.0
    @test char_coverage("xyz", vocab) ≈ 0.0
    @test char_coverage("lx", vocab) ≈ 0.5
    @test char_coverage("", vocab) == 0.0
end

@testset "unknown_characters" begin
    vocab = Set(["l", "o", "w", "</w>"])
    @test unknown_characters("low", vocab) == Set{Char}()
    @test unknown_characters("lowx", vocab) == Set([x])
    @test unknown_characters("xyz", vocab) == Set([x, y, z])
    @test unknown_characters("", vocab) == Set{Char}()
end

@testset "encode_truncated" begin
    corpus = "low low low lower lower lowest"
    t = train_tokenizer(corpus, 10)
    full_ids = encode(t, "low lower lowest")
    trunc_ids = encode_truncated(t, "low lower lowest", 2)
    @test length(trunc_ids) == 2
    @test trunc_ids == full_ids[1:2]
    # no truncation needed
    short_ids = encode_truncated(t, "low", 100)
    @test short_ids == encode(t, "low")
end

@testset "remove_special_tokens_from" begin
    tokens = ["<bos>", "lo", "w", "</w>", "<eos>"]
    filtered = remove_special_tokens_from(tokens, ["<bos>", "<eos>"])
    @test filtered == ["lo", "w", "</w>"]
    @test remove_special_tokens_from(String[], ["<pad>"]) == String[]
end

@testset "token_count" begin
    corpus = "low low low lower lower lowest"
    _, merges = train_bpe(corpus, 10)
    @test token_count("low lower", merges) == length(encode_text("low lower", merges))
    @test token_count("", merges) == 0
end

@testset "encode_parallel" begin
    corpus = "low low low lower lower lowest"
    t = train_tokenizer(corpus, 10)
    texts = ["low", "lower", "lowest"]
    par = encode_parallel(t, texts)
    seq = [encode(t, text) for text in texts]
    @test par == seq
    @test encode_parallel(t, String[]) == Vector{Int}[]
end

@testset "prefix_tokens and suffix_tokens" begin
    vocab = Set(["low", "lower", "lowest", "high", "lo", "</w>"])
    @test prefix_tokens(vocab, "lo") == Set(["low", "lower", "lowest", "lo"])
    @test suffix_tokens(vocab, "est") == Set(["lowest"])
    @test prefix_tokens(vocab, "xyz") == Set{String}()
    @test suffix_tokens(vocab, "xyz") == Set{String}()
end

@testset "is_special_token" begin
    specials = ["<unk>", "<pad>", "<bos>"]
    @test is_special_token("<unk>", specials) == true
    @test is_special_token("hello", specials) == false
    @test is_special_token("<pad>", specials) == true
end

@testset "common_prefixes" begin
    vocab = Set(["low", "lower", "lowest", "high", "higher"])
    prefixes = common_prefixes(vocab)
    @test prefixes["lo"] >= 3
    @test prefixes["hi"] >= 2
    @test common_prefixes(Set{String}()) == Dict{String,Int}()
end

@testset "average_compression" begin
    corpus = "low low low lower lower lowest"
    _, merges = train_bpe(corpus, 10)
    avg = average_compression(["low lower", "lowest"], merges)
    @test avg > 0.0
    @test average_compression(String[], merges) == 0.0
end

@testset "batch_coverage" begin
    corpus = "low low low lower lower lowest"
    _, merges = train_bpe(corpus, 10)
    covs = batch_coverage(["low lower", "xyz abc"], merges)
    @test length(covs) == 2
    @test covs[1] > covs[2]
end

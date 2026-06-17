module BytePairEncoding

export normalize_unicode,
    is_valid_utf8,
    word_to_graphemes,
    word_to_symbols,
    count_word_frequencies,
    initialize_word_symbols,
    count_pairs,
    best_pair,
    merge_symbols,
    train_bpe,
    get_vocabulary,
    preprocess_text,
    load_corpus,
    encode_word,
    encode_text,
    decode_tokens,
    save_merges,
    load_merges,
    save_vocab,
    compression_ratio,
    token_frequencies,
    vocab_size_history,
    add_special_tokens,
    encode_batch,
    encode_word_with_dropout,
    build_vocab_index,
    tokens_to_ids,
    ids_to_tokens,
    save_vocab_index,
    load_vocab_index,
    pad_sequence,
    truncate_sequence,
    prepare_batch,
    pretokenize,
    count_frequencies_pretokenized,
    tokenize,
    BPETokenizer,
    train_tokenizer,
    encode,
    decode,
    save_tokenizer,
    load_tokenizer,
    most_common_tokens,
    average_token_length,
    coverage,
    text_to_bytes,
    bytes_to_text,
    train_byte_bpe,
    encode_byte_level,
    validate_merges,
    validate_vocab_index,
    validate_tokenizer,
    train_bpe_protected,
    encode_with_protected_tokens,
    count_word_frequencies_streaming,
    train_bpe_streaming,
    wordpiece_tokenize,
    train_wordpiece,
    token_length_distribution,
    subword_fertility,
    vocab_overlap,
    MergeRecord,
    train_bpe_with_history,
    format_merge_history,
    TokenizerConfig,
    save_config,
    load_config,
    train_from_config


using Unicode

"""
    normalize_unicode(text; form=:NFC) → String

Normalize Unicode text to the specified form (:NFC, :NFD, :NFKC, :NFKD).
NFC is the default and recommended form for tokenizer input.
"""
function normalize_unicode(text::String; form::Symbol=:NFC)::String
    return Unicode.normalize(text, form)
end


"""
    is_valid_utf8(data) → Bool

Check whether a byte vector or string contains valid UTF-8.
"""
function is_valid_utf8(data::Vector{UInt8})::Bool
    return isvalid(String, data)
end

function is_valid_utf8(text::String)::Bool
    return isvalid(text)
end


"""
    preprocess_text(text; lowercase=true)

Normalize text for BPE training: optionally lowercase and collapse whitespace.

Example:
"  Hello   World  " -> "hello world"
"""
function preprocess_text(text::String; lowercase::Bool=true)::String
    result = lowercase ? Base.lowercase(text) : text
    result = replace(result, r"\s+" => " ")
    return strip(result) |> String
end


"""
    load_corpus(filepath)

Read a text corpus from a file and return it as a trimmed string.

Raises an error if the file does not exist.
"""
function load_corpus(filepath::String)::String
    if !isfile(filepath)
        error("corpus file not found: $filepath")
    end
    return strip(read(filepath, String)) |> String
end


"""
    word_to_graphemes(word) → Vector{String}

Convert a word into grapheme clusters with an end-of-word marker.
Unlike `word_to_symbols`, this correctly handles multi-codepoint characters
like emoji (👨‍👩‍👧) and accented characters as single units.
"""
function word_to_graphemes(word::String)::Vector{String}
    graphemes_list = [String(g) for g in graphemes(word)]
    push!(graphemes_list, "</w>")
    return graphemes_list
end


"""
    word_to_symbols(word)

Convert a word into character-level symbols and append an end-of-word marker.

Example:
"low" -> ["l", "o", "w", "</w>"]
"""
function word_to_symbols(word::String)::Vector{String}
    symbols = string.(collect(word))
    push!(symbols, "</w>")

    return symbols
end


"""
    count_word_frequencies(corpus)

Count how many times each word appears in the corpus.

Example:
"low lower lowest lower" -> Dict("low" => 1, "lower" => 2, "lowest" => 1)
"""
function count_word_frequencies(corpus::String)::Dict{String,Int}
    frequencies = Dict{String,Int}()

    for word in split(corpus)
        current_count = get(frequencies, word, 0)
        frequencies[word] = current_count + 1
    end

    return frequencies
end


"""
    initialize_word_symbols(word_frequencies)
Convert each unique word into its initial symbol representation.

Example:
Dict("low" => 1) -> Dict(["l", "o", "w", "</w>"] => 1)
"""
function initialize_word_symbols(word_frequencies::Dict{String,Int})::Dict{Vector{String},Int}
    word_symbols = Dict{Vector{String},Int}()

    for (word, frequency) in word_frequencies
        symbols = word_to_symbols(word)
        word_symbols[symbols] = frequency
    end
    return word_symbols
end


"""
    count_pairs(word_symbols)

Count adjacent symbol pairs across all tokenized words.

Word frequency is used, so repeated words influence pair counts.

Example:
["l", "o", "w", "</w>"] => 2

Pairs:
("l", "o") => 2
("o", "w") => 2
("w", "</w>") => 2
"""

function count_pairs(
    word_symbols::Dict{Vector{String},Int},
)::Dict{Tuple{String,String},Int}
    pair_counts = Dict{Tuple{String,String},Int}()

    for (symbols, frequency) in word_symbols
        if length(symbols) < 2
            continue
        end

        for index in 1:(length(symbols)-1)
            pair = (symbols[index], symbols[index+1])
            current_count = get(pair_counts, pair, 0)
            pair_counts[pair] = current_count + frequency
        end
    end
    return pair_counts
end


"""
    best_pair(pair_counts)

Return the most frequent adjacent pair.

If there are no pairs, return 'nothing'
"""

function best_pair(
    pair_counts::Dict{Tuple{String,String},Int},
)::Union{Tuple{String,String},Nothing}
    if isempty(pair_counts)
        return nothing
    end

    best = nothing
    best_count = -1

    for (pair, count) in pair_counts
        if count > best_count
            best = pair
            best_count = count
        end
    end

    return best

end


function merge_symbols(
    symbols::Vector{String},
    pair::Tuple{String,String},
)::Vector{String}
    merged_symbols = String[]
    index = 1

    while index <= length(symbols)
        has_next_symbol = index < length(symbols)
        should_merge = has_next_symbol &&
            symbols[index] == pair[1] &&
            symbols[index + 1] == pair[2]

        if should_merge
            push!(merged_symbols, pair[1] * pair[2])
            index += 2
            continue
        end

        push!(merged_symbols, symbols[index])
        index += 1
    end
    return merged_symbols
end


"""
    train_bpe(corpus, num_merges; verbose=false, min_frequency=0)

Run the full BPE training loop for a given number of merges.

Options:
- `verbose=true`: print each merge step as it happens
- `min_frequency`: stop merging when the best pair frequency drops below this threshold

Returns the final vocabulary (word_symbols) and the list of merges performed.
"""
function train_bpe(corpus::String, num_merges::Int; verbose::Bool=false, min_frequency::Int=0)::Tuple{Dict{Vector{String},Int},Vector{Tuple{String,String}}}
    frequencies = count_word_frequencies(corpus)
    word_symbols = initialize_word_symbols(frequencies)
    merges = Tuple{String,String}[]

    for i in 1:num_merges
        pair_counts = count_pairs(word_symbols)
        pair = best_pair(pair_counts)

        if pair === nothing
            verbose && println("stopping early: no more pairs at step $i")
            break
        end

        if min_frequency > 0 && pair_counts[pair] < min_frequency
            verbose && println("stopping early: best pair frequency $(pair_counts[pair]) < min_frequency $min_frequency at step $i")
            break
        end

        if verbose
            println("merge $i: $(pair[1]) + $(pair[2]) -> $(pair[1])$(pair[2]) (freq=$(pair_counts[pair]))")
        end

        push!(merges, pair)

        new_word_symbols = Dict{Vector{String},Int}()
        for (symbols, freq) in word_symbols
            new_word_symbols[merge_symbols(symbols, pair)] = freq
        end
        word_symbols = new_word_symbols
    end

    return (word_symbols, merges)
end


"""
    get_vocabulary(word_symbols)

Extract the set of unique tokens from the trained vocabulary.
"""
function get_vocabulary(word_symbols::Dict{Vector{String},Int})::Set{String}
    vocab = Set{String}()
    for (symbols, _) in word_symbols
        for symbol in symbols
            push!(vocab, symbol)
        end
    end
    return vocab
end


"""
    encode_word(word, merges)

Apply learned BPE merges to tokenize a single word.

Returns the token sequence after applying all merges in order.

Example:
    merges = [("l", "o"), ("lo", "w")]
    encode_word("low", merges) -> ["low", "</w>"]
"""
function encode_word(word::String, merges::Vector{Tuple{String,String}})::Vector{String}
    symbols = word_to_symbols(word)
    for merge in merges
        symbols = merge_symbols(symbols, merge)
    end
    return symbols
end


"""
    encode_text(text, merges)

Tokenize a full text string using learned BPE merges.

Splits text into words, encodes each word, and returns the flat token list.
"""
function encode_text(text::String, merges::Vector{Tuple{String,String}})::Vector{String}
    tokens = String[]
    for word in split(text)
        append!(tokens, encode_word(String(word), merges))
    end
    return tokens
end


"""
    decode_tokens(tokens)

Reconstruct text from a sequence of BPE tokens.

Joins tokens and converts end-of-word markers back to spaces.

Example:
    decode_tokens(["low", "</w>", "er", "</w>"]) -> "low er"
"""
function decode_tokens(tokens::Vector{String})::String
    text = join(tokens, "")
    text = replace(text, "</w>" => " ")
    return strip(text) |> String
end


"""
    save_merges(merges, filepath)

Write BPE merge rules to a tab-separated file, one merge per line.
"""
function save_merges(merges::Vector{Tuple{String,String}}, filepath::String)
    open(filepath, "w") do io
        for (a, b) in merges
            println(io, a, "\t", b)
        end
    end
end


"""
    load_merges(filepath)

Read BPE merge rules from a tab-separated file.

Raises an error if the file does not exist.
"""
function load_merges(filepath::String)::Vector{Tuple{String,String}}
    if !isfile(filepath)
        error("merges file not found: $filepath")
    end
    merges = Tuple{String,String}[]
    for line in eachline(filepath)
        parts = split(line, "\t")
        if length(parts) == 2
            push!(merges, (String(parts[1]), String(parts[2])))
        end
    end
    return merges
end


"""
    compression_ratio(original_text, tokens)

Compute the ratio of original character count to token count.

Higher values indicate better compression (fewer tokens per character).
"""
function compression_ratio(original_text::String, tokens::Vector{String})::Float64
    num_chars = length(original_text)
    num_tokens = length(tokens)
    if num_tokens == 0
        return 0.0
    end
    return num_chars / num_tokens
end


"""
    vocab_size_history(corpus, num_merges)

Track how vocabulary size changes at each merge step during training.

Returns a vector of vocabulary sizes, starting from the initial character-level vocab.
"""
function vocab_size_history(corpus::String, num_merges::Int)::Vector{Int}
    frequencies = count_word_frequencies(corpus)
    word_symbols = initialize_word_symbols(frequencies)
    history = Int[length(get_vocabulary(word_symbols))]

    for _ in 1:num_merges
        pair_counts = count_pairs(word_symbols)
        pair = best_pair(pair_counts)
        if pair === nothing
            break
        end
        new_word_symbols = Dict{Vector{String},Int}()
        for (symbols, freq) in word_symbols
            new_word_symbols[merge_symbols(symbols, pair)] = freq
        end
        word_symbols = new_word_symbols
        push!(history, length(get_vocabulary(word_symbols)))
    end
    return history
end


"""
    token_frequencies(tokens)

Count the frequency of each token in a token sequence.

Returns a Dict mapping tokens to their counts.
"""
function token_frequencies(tokens::Vector{String})::Dict{String,Int}
    freqs = Dict{String,Int}()
    for token in tokens
        freqs[token] = get(freqs, token, 0) + 1
    end
    return freqs
end


"""
    save_vocab(vocab, filepath)

Write vocabulary tokens to a file, one token per line, sorted alphabetically.
"""
function save_vocab(vocab::Set{String}, filepath::String)
    open(filepath, "w") do io
        for token in sort(collect(vocab))
            println(io, token)
        end
    end
end


"""
    add_special_tokens(vocab, special)

Add special tokens (e.g. "<unk>", "<pad>", "<bos>", "<eos>") to a vocabulary set.

Returns a new set with the special tokens included.
"""
function add_special_tokens(vocab::Set{String}, special::Vector{String})::Set{String}
    new_vocab = copy(vocab)
    for token in special
        push!(new_vocab, token)
    end
    return new_vocab
end


"""
    encode_batch(texts, merges)

Encode multiple text strings using learned BPE merges.

Returns a vector of token sequences, one per input text.
"""
function encode_batch(texts::Vector{String}, merges::Vector{Tuple{String,String}})::Vector{Vector{String}}
    return [encode_text(text, merges) for text in texts]
end


"""
    encode_word_with_dropout(word, merges; dropout=0.1)

Apply BPE merges with stochastic dropout for subword regularization.

Each merge is skipped with probability `dropout`, producing varied tokenizations
of the same word. Useful for training robustness.

With `dropout=0.0`, behaves identically to `encode_word`.
"""
function encode_word_with_dropout(word::String, merges::Vector{Tuple{String,String}}; dropout::Float64=0.1)::Vector{String}
    symbols = word_to_symbols(word)
    for merge in merges
        if dropout > 0.0 && rand() < dropout
            continue
        end
        symbols = merge_symbols(symbols, merge)
    end
    return symbols
end


"""
    build_vocab_index(vocab, special_tokens) → Dict{String,Int}

Assign integer IDs to tokens. Special tokens are assigned first (starting at 1),
followed by vocabulary tokens in sorted order.
"""
function build_vocab_index(vocab::Set{String}, special_tokens::Vector{String}=String[])::Dict{String,Int}
    index = Dict{String,Int}()
    id = 1
    for token in special_tokens
        index[token] = id
        id += 1
    end
    for token in sort(collect(vocab))
        if !haskey(index, token)
            index[token] = id
            id += 1
        end
    end
    return index
end


"""
    tokens_to_ids(tokens, index; unk_id=0) → Vector{Int}

Map a sequence of string tokens to integer IDs using a vocabulary index.
Unknown tokens are mapped to `unk_id`.
"""
function tokens_to_ids(tokens::Vector{String}, index::Dict{String,Int}; unk_id::Int=0)::Vector{Int}
    return [get(index, token, unk_id) for token in tokens]
end


"""
    ids_to_tokens(ids, index) → Vector{String}

Reverse-map integer IDs back to string tokens using a vocabulary index.
Unknown IDs are mapped to "<unk>".
"""
function ids_to_tokens(ids::Vector{Int}, index::Dict{String,Int})::Vector{String}
    reverse_index = Dict{Int,String}(v => k for (k, v) in index)
    return [get(reverse_index, id, "<unk>") for id in ids]
end


"""
    save_vocab_index(index, filepath)

Write a vocabulary index to a tab-separated file (token<TAB>id), sorted by ID.
"""
function save_vocab_index(index::Dict{String,Int}, filepath::String)
    sorted = sort(collect(index), by=x -> x[2])
    open(filepath, "w") do io
        for (token, id) in sorted
            println(io, token, "\t", id)
        end
    end
end


"""
    load_vocab_index(filepath) → Dict{String,Int}

Read a vocabulary index from a tab-separated file.

Raises an error if the file does not exist.
"""
function load_vocab_index(filepath::String)::Dict{String,Int}
    if !isfile(filepath)
        error("vocab index file not found: $filepath")
    end
    index = Dict{String,Int}()
    for line in eachline(filepath)
        parts = split(line, "\t")
        if length(parts) == 2
            index[String(parts[1])] = parse(Int, parts[2])
        end
    end
    return index
end


"""
    pad_sequence(ids, max_len; pad_id=0) → Vector{Int}

Right-pad a sequence of integer IDs to a fixed length.

If the sequence is already longer than `max_len`, it is returned unchanged.
"""
function pad_sequence(ids::Vector{Int}, max_len::Int; pad_id::Int=0)::Vector{Int}
    current_len = length(ids)
    if current_len >= max_len
        return copy(ids)
    end
    return vcat(ids, fill(pad_id, max_len - current_len))
end


"""
    truncate_sequence(ids, max_len) → Vector{Int}

Truncate a sequence of integer IDs to at most `max_len` elements from the left.

If the sequence is already shorter than or equal to `max_len`, it is returned unchanged.
"""
function truncate_sequence(ids::Vector{Int}, max_len::Int)::Vector{Int}
    if length(ids) <= max_len
        return copy(ids)
    end
    return ids[1:max_len]
end


"""
    prepare_batch(batch, max_len; pad_id=0) → Vector{Vector{Int}}

Truncate and pad a batch of ID sequences to uniform length.

Each sequence is first truncated to `max_len`, then right-padded with `pad_id`.
"""
function prepare_batch(batch::Vector{Vector{Int}}, max_len::Int; pad_id::Int=0)::Vector{Vector{Int}}
    return [pad_sequence(truncate_sequence(seq, max_len), max_len, pad_id=pad_id) for seq in batch]
end


const GPT2_PATTERN = r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"

"""
    pretokenize(text; pattern=GPT2_PATTERN) → Vector{String}

Split text into chunks using a regex pattern (GPT-2-style by default).

Each match becomes a separate chunk for downstream BPE encoding.
"""
function pretokenize(text::String; pattern::Regex=GPT2_PATTERN)::Vector{String}
    return [String(m.match) for m in eachmatch(pattern, text)]
end


"""
    count_frequencies_pretokenized(text; pattern=GPT2_PATTERN) → Dict{String,Int}

Count word frequencies using regex pre-tokenization instead of simple whitespace splitting.
"""
function count_frequencies_pretokenized(text::String; pattern::Regex=GPT2_PATTERN)::Dict{String,Int}
    chunks = pretokenize(text, pattern=pattern)
    frequencies = Dict{String,Int}()
    for chunk in chunks
        frequencies[chunk] = get(frequencies, chunk, 0) + 1
    end
    return frequencies
end


"""
    tokenize(text, merges; pattern=GPT2_PATTERN) → Vector{String}

High-level end-to-end tokenization: preprocess → pretokenize → encode each chunk → flat token list.

Combines regex pre-tokenization with BPE encoding for a complete pipeline.
"""
function tokenize(text::String, merges::Vector{Tuple{String,String}}; pattern::Regex=GPT2_PATTERN)::Vector{String}
    processed = preprocess_text(text)
    chunks = pretokenize(processed, pattern=pattern)
    tokens = String[]
    for chunk in chunks
        word = strip(chunk) |> String
        if isempty(word)
            continue
        end
        append!(tokens, encode_word(word, merges))
    end
    return tokens
end


"""
    BPETokenizer

A complete BPE tokenizer bundling trained merges, vocabulary, token-to-ID index,
ID-to-token reverse index, and special tokens into a single struct.
"""
struct BPETokenizer
    merges::Vector{Tuple{String,String}}
    vocab::Set{String}
    vocab_index::Dict{String,Int}
    id_to_token::Dict{Int,String}
    special_tokens::Vector{String}
end


"""
    train_tokenizer(corpus, num_merges; special_tokens=["<unk>", "<pad>"], verbose=false, min_frequency=0) → BPETokenizer

Train a complete BPE tokenizer from a text corpus.

Returns a `BPETokenizer` with trained merges, vocabulary, and token-to-ID mappings.
"""
function train_tokenizer(
    corpus::String,
    num_merges::Int;
    special_tokens::Vector{String}=["<unk>", "<pad>"],
    verbose::Bool=false,
    min_frequency::Int=0
)::BPETokenizer
    processed = preprocess_text(corpus)
    word_symbols, merges = train_bpe(processed, num_merges, verbose=verbose, min_frequency=min_frequency)
    vocab = get_vocabulary(word_symbols)
    vocab = add_special_tokens(vocab, special_tokens)
    vocab_index = build_vocab_index(vocab, special_tokens)
    id_to_token = Dict{Int,String}(v => k for (k, v) in vocab_index)
    return BPETokenizer(merges, vocab, vocab_index, id_to_token, special_tokens)
end


"""
    encode(t::BPETokenizer, text) → Vector{Int}

Tokenize text and convert to integer IDs using the tokenizer's vocabulary.

Unknown tokens are mapped to the ID of "<unk>" if present, otherwise 0.
"""
function encode(t::BPETokenizer, text::String)::Vector{Int}
    string_tokens = tokenize(text, t.merges)
    unk_id = get(t.vocab_index, "<unk>", 0)
    return tokens_to_ids(string_tokens, t.vocab_index, unk_id=unk_id)
end


"""
    decode(t::BPETokenizer, ids) → String

Convert integer IDs back to text using the tokenizer's reverse index.
"""
function decode(t::BPETokenizer, ids::Vector{Int})::String
    string_tokens = [get(t.id_to_token, id, "<unk>") for id in ids]
    return decode_tokens(string_tokens)
end


"""
    save_tokenizer(t::BPETokenizer, dir)

Save all tokenizer state to a directory: merges.tsv, vocab_index.tsv, and special_tokens.txt.

Creates the directory if it does not exist.
"""
function save_tokenizer(t::BPETokenizer, dir::String)
    mkpath(dir)
    save_merges(t.merges, joinpath(dir, "merges.tsv"))
    save_vocab_index(t.vocab_index, joinpath(dir, "vocab_index.tsv"))
    open(joinpath(dir, "special_tokens.txt"), "w") do io
        for token in t.special_tokens
            println(io, token)
        end
    end
end


"""
    load_tokenizer(dir) → BPETokenizer

Load a tokenizer from a directory previously saved by `save_tokenizer`.

Raises an error if the directory does not exist.
"""
function load_tokenizer(dir::String)::BPETokenizer
    if !isdir(dir)
        error("tokenizer directory not found: $dir")
    end
    merges = load_merges(joinpath(dir, "merges.tsv"))
    vocab_index = load_vocab_index(joinpath(dir, "vocab_index.tsv"))

    special_tokens_path = joinpath(dir, "special_tokens.txt")
    special_tokens = if isfile(special_tokens_path)
        filter(!isempty, readlines(special_tokens_path))
    else
        String[]
    end

    vocab = Set(keys(vocab_index))
    id_to_token = Dict{Int,String}(v => k for (k, v) in vocab_index)
    t = BPETokenizer(merges, vocab, vocab_index, id_to_token, special_tokens)
    warnings = validate_tokenizer(t)
    for w in warnings
        @warn "tokenizer validation: $w"
    end
    return t
end


"""
    most_common_tokens(tokens, n) → Vector{Tuple{String,Int}}

Return the top-N most frequent tokens from a token sequence, sorted by descending frequency.
"""
function most_common_tokens(tokens::Vector{String}, n::Int)::Vector{Tuple{String,Int}}
    freqs = token_frequencies(tokens)
    sorted = sort(collect(freqs), by=x -> -x[2])
    top_n = sorted[1:min(n, length(sorted))]
    return [(k, v) for (k, v) in top_n]
end


"""
    average_token_length(vocab) → Float64

Compute the mean character count of tokens in a vocabulary set.

Returns 0.0 for an empty vocabulary.
"""
function average_token_length(vocab::Set{String})::Float64
    if isempty(vocab)
        return 0.0
    end
    return sum(length(t) for t in vocab) / length(vocab)
end


"""
    coverage(text, merges) → Float64

Measure vocabulary completeness: the fraction of words in `text` that are fully
encodable without producing single-character tokens (beyond the end-of-word marker).

A word is considered "covered" if all its BPE tokens are multi-character or the
word itself is a single character. Returns 0.0 for empty text.
"""
function coverage(text::String, merges::Vector{Tuple{String,String}})::Float64
    words = split(text)
    if isempty(words)
        return 0.0
    end
    covered = 0
    for word in words
        tokens = encode_word(String(word), merges)
        # A word is covered if no token is a single character (excluding </w>)
        non_marker = filter(t -> t != "</w>", tokens)
        if all(t -> length(t) > 1 || length(String(word)) == 1, non_marker)
            covered += 1
        end
    end
    return covered / length(words)
end


"""
    text_to_bytes(text) → Vector{String}

Convert text to a sequence of hex byte strings (e.g. 'L' → "4c").

Each byte of the UTF-8 encoding becomes a two-character lowercase hex string.
"""
function text_to_bytes(text::String)::Vector{String}
    return [string(b, base=16, pad=2) for b in Vector{UInt8}(codeunits(text))]
end


"""
    bytes_to_text(byte_tokens) → String

Reconstruct a string from a sequence of hex byte tokens.

Each token is parsed as a sequence of hex byte pairs and converted back to characters.
For example, ["4c6f", "77"] → "Low".
"""
function bytes_to_text(byte_tokens::Vector{String})::String
    bytes = UInt8[]
    for token in byte_tokens
        for i in 1:2:length(token)
            push!(bytes, parse(UInt8, token[i:i+1], base=16))
        end
    end
    return String(bytes)
end


"""
    train_byte_bpe(text, num_merges; verbose=false) → Tuple{Dict{Vector{String},Int}, Vector{Tuple{String,String}}}

Train BPE on byte-level representations instead of characters.

Text is first converted to hex byte sequences, then standard BPE training is applied.
Returns word_symbols and merges in the byte domain.
"""
function train_byte_bpe(text::String, num_merges::Int; verbose::Bool=false)::Tuple{Dict{Vector{String},Int},Vector{Tuple{String,String}}}
    words = split(text)
    word_symbols = Dict{Vector{String},Int}()
    for word in words
        byte_seq = text_to_bytes(String(word))
        word_symbols[byte_seq] = get(word_symbols, byte_seq, 0) + 1
    end

    merges = Tuple{String,String}[]
    for i in 1:num_merges
        pair_counts = count_pairs(word_symbols)
        pair = best_pair(pair_counts)
        if pair === nothing
            verbose && println("stopping early: no more pairs at step $i")
            break
        end
        if verbose
            println("merge $i: $(pair[1]) + $(pair[2]) -> $(pair[1])$(pair[2]) (freq=$(pair_counts[pair]))")
        end
        push!(merges, pair)
        new_word_symbols = Dict{Vector{String},Int}()
        for (symbols, freq) in word_symbols
            new_word_symbols[merge_symbols(symbols, pair)] = freq
        end
        word_symbols = new_word_symbols
    end
    return (word_symbols, merges)
end


"""
    encode_byte_level(text, merges) → Vector{String}

Apply byte-level BPE merges to encode text.

Text is converted to hex bytes, then merges are applied to produce byte-level tokens.
"""
function encode_byte_level(text::String, merges::Vector{Tuple{String,String}})::Vector{String}
    tokens = String[]
    for word in split(text)
        symbols = text_to_bytes(String(word))
        for merge in merges
            symbols = merge_symbols(symbols, merge)
        end
        append!(tokens, symbols)
    end
    return tokens
end


"""
    train_bpe_protected(corpus, num_merges; never_merge=Set(), verbose=false, min_frequency=0)

Train BPE with a set of pairs that should never be merged.
Pairs in `never_merge` are skipped during training even if they are the most frequent.
"""
function train_bpe_protected(
    corpus::String,
    num_merges::Int;
    never_merge::Set{Tuple{String,String}}=Set{Tuple{String,String}}(),
    verbose::Bool=false,
    min_frequency::Int=0
)::Tuple{Dict{Vector{String},Int},Vector{Tuple{String,String}}}
    frequencies = count_word_frequencies(corpus)
    word_symbols = initialize_word_symbols(frequencies)
    merges = Tuple{String,String}[]

    for i in 1:num_merges
        pair_counts = count_pairs(word_symbols)
        # filter out protected pairs
        for p in never_merge
            delete!(pair_counts, p)
        end
        pair = best_pair(pair_counts)

        if pair === nothing
            verbose && println("stopping early: no more pairs at step $i")
            break
        end
        if min_frequency > 0 && pair_counts[pair] < min_frequency
            verbose && println("stopping early: best pair frequency $(pair_counts[pair]) < min_frequency $min_frequency at step $i")
            break
        end
        if verbose
            println("merge $i: $(pair[1]) + $(pair[2]) -> $(pair[1])$(pair[2]) (freq=$(pair_counts[pair]))")
        end
        push!(merges, pair)
        new_word_symbols = Dict{Vector{String},Int}()
        for (symbols, freq) in word_symbols
            new_word_symbols[merge_symbols(symbols, pair)] = freq
        end
        word_symbols = new_word_symbols
    end
    return (word_symbols, merges)
end


"""
    TokenizerConfig

Configuration struct for training parameters.
"""
struct TokenizerConfig
    num_merges::Int
    min_frequency::Int
    special_tokens::Vector{String}
    lowercase::Bool
    verbose::Bool
end

TokenizerConfig(;
    num_merges::Int=1000,
    min_frequency::Int=0,
    special_tokens::Vector{String}=["<unk>", "<pad>"],
    lowercase::Bool=true,
    verbose::Bool=false
) = TokenizerConfig(num_merges, min_frequency, special_tokens, lowercase, verbose)


"""
    save_config(config, filepath)

Save a TokenizerConfig to a JSON file.
"""
function save_config(config::TokenizerConfig, filepath::String)
    open(filepath, "w") do io
        println(io, "{")
        println(io, "  \"num_merges\": $(config.num_merges),")
        println(io, "  \"min_frequency\": $(config.min_frequency),")
        println(io, "  \"special_tokens\": [$(join(["\"$t\"" for t in config.special_tokens], ", "))],")
        println(io, "  \"lowercase\": $(config.lowercase),")
        println(io, "  \"verbose\": $(config.verbose)")
        println(io, "}")
    end
end


"""
    load_config(filepath) → TokenizerConfig

Load a TokenizerConfig from a JSON file. Uses simple parsing (no JSON dependency).
"""
function load_config(filepath::String)::TokenizerConfig
    if !isfile(filepath)
        error("config file not found: $filepath")
    end
    text = read(filepath, String)
    # simple JSON parsing for flat config
    get_int(key) = parse(Int, match(Regex("\"$key\":\\s*(\\d+)"), text).captures[1])
    get_bool(key) = match(Regex("\"$key\":\\s*(true|false)"), text).captures[1] == "true"
    function get_string_array(key)
        m = match(Regex("\"$key\":\\s*\\[([^\\]]*)\\]"), text)
        m === nothing && return String[]
        return [String(s.captures[1]) for s in eachmatch(r"\"([^\"]+)\"", m.captures[1])]
    end
    return TokenizerConfig(
        num_merges=get_int("num_merges"),
        min_frequency=get_int("min_frequency"),
        special_tokens=get_string_array("special_tokens"),
        lowercase=get_bool("lowercase"),
        verbose=get_bool("verbose")
    )
end


"""
    train_from_config(corpus, config) → BPETokenizer

Train a tokenizer using parameters from a TokenizerConfig.
"""
function train_from_config(corpus::String, config::TokenizerConfig)::BPETokenizer
    processed = config.lowercase ? preprocess_text(corpus) : preprocess_text(corpus, lowercase=false)
    return train_tokenizer(processed, config.num_merges,
        special_tokens=config.special_tokens,
        verbose=config.verbose,
        min_frequency=config.min_frequency)
end


"""
    MergeRecord

A record of a single merge step during BPE training.
"""
struct MergeRecord
    step::Int
    pair::Tuple{String,String}
    frequency::Int
    new_token::String
    vocab_size::Int
end


"""
    train_bpe_with_history(corpus, num_merges; verbose=false, min_frequency=0)

Train BPE and return a full merge history alongside the standard outputs.
Returns (word_symbols, merges, history::Vector{MergeRecord}).
"""
function train_bpe_with_history(
    corpus::String,
    num_merges::Int;
    verbose::Bool=false,
    min_frequency::Int=0
)::Tuple{Dict{Vector{String},Int},Vector{Tuple{String,String}},Vector{MergeRecord}}
    frequencies = count_word_frequencies(corpus)
    word_symbols = initialize_word_symbols(frequencies)
    merges = Tuple{String,String}[]
    history = MergeRecord[]

    for i in 1:num_merges
        pair_counts = count_pairs(word_symbols)
        pair = best_pair(pair_counts)
        if pair === nothing
            break
        end
        freq = pair_counts[pair]
        if min_frequency > 0 && freq < min_frequency
            break
        end
        if verbose
            println("merge $i: $(pair[1]) + $(pair[2]) -> $(pair[1])$(pair[2]) (freq=$freq)")
        end
        push!(merges, pair)
        new_word_symbols = Dict{Vector{String},Int}()
        for (symbols, f) in word_symbols
            new_word_symbols[merge_symbols(symbols, pair)] = f
        end
        word_symbols = new_word_symbols
        vs = length(get_vocabulary(word_symbols))
        push!(history, MergeRecord(i, pair, freq, pair[1] * pair[2], vs))
    end
    return (word_symbols, merges, history)
end


"""
    format_merge_history(history) → String

Format merge history as a readable table.
"""
function format_merge_history(history::Vector{MergeRecord})::String
    lines = ["Step | Pair            | Freq | New Token   | Vocab Size"]
    push!(lines, "-" ^ 60)
    for r in history
        pair_str = "$(r.pair[1]) + $(r.pair[2])"
        line = "$(lpad(r.step, 4)) | $(rpad(pair_str, 15)) | $(lpad(r.frequency, 4)) | $(rpad(r.new_token, 11)) | $(lpad(r.vocab_size, 10))"
        push!(lines, line)
    end
    return join(lines, "\n")
end


"""
    token_length_distribution(vocab) → Dict{Int,Int}

Compute a histogram of token lengths in a vocabulary.
Returns a Dict mapping character count to number of tokens with that length.
"""
function token_length_distribution(vocab::Set{String})::Dict{Int,Int}
    dist = Dict{Int,Int}()
    for token in vocab
        len = length(token)
        dist[len] = get(dist, len, 0) + 1
    end
    return dist
end


"""
    subword_fertility(text, merges) → Float64

Measure the average number of subword tokens per original word.
Lower values indicate better compression / vocabulary coverage.
"""
function subword_fertility(text::String, merges::Vector{Tuple{String,String}})::Float64
    words = split(text)
    if isempty(words)
        return 0.0
    end
    total_tokens = 0
    for word in words
        tokens = encode_word(String(word), merges)
        total_tokens += length(tokens)
    end
    return total_tokens / length(words)
end


"""
    vocab_overlap(vocab1, vocab2) → NamedTuple{(:jaccard, :shared, :only1, :only2)}

Compare two vocabularies and compute overlap statistics.
Returns Jaccard similarity and the sets of shared/unique tokens.
"""
function vocab_overlap(vocab1::Set{String}, vocab2::Set{String})
    shared = intersect(vocab1, vocab2)
    only1 = setdiff(vocab1, vocab2)
    only2 = setdiff(vocab2, vocab1)
    union_size = length(union(vocab1, vocab2))
    jaccard = union_size == 0 ? 0.0 : length(shared) / union_size
    return (jaccard=jaccard, shared=shared, only1=only1, only2=only2)
end


"""
    train_wordpiece(corpus, vocab_size; min_frequency=2) → Set{String}

Build a WordPiece vocabulary from a corpus. Starts with character-level tokens
and iteratively adds the most frequent subword that improves coverage.
Returns the vocabulary set (including ## prefixed continuation tokens).
"""
function train_wordpiece(corpus::String, vocab_size::Int; min_frequency::Int=2)::Set{String}
    words = split(preprocess_text(corpus))
    word_freqs = Dict{String,Int}()
    for w in words
        word_freqs[String(w)] = get(word_freqs, String(w), 0) + 1
    end

    # start with all single characters
    vocab = Set{String}()
    for word in keys(word_freqs)
        for (i, ch) in enumerate(word)
            token = i == 1 ? string(ch) : "##" * string(ch)
            push!(vocab, token)
        end
    end

    # iteratively add most frequent pairs as merged tokens
    while length(vocab) < vocab_size
        pair_scores = Dict{String,Int}()
        for (word, freq) in word_freqs
            freq < min_frequency && continue
            chars = collect(word)
            tokens = String[]
            for (i, ch) in enumerate(chars)
                push!(tokens, i == 1 ? string(ch) : "##" * string(ch))
            end
            # try merging adjacent token pairs
            for j in 1:length(tokens)-1
                merged = tokens[j] * replace(tokens[j+1], "##" => "")
                if !(merged in vocab)
                    pair_scores[merged] = get(pair_scores, merged, 0) + freq
                end
            end
        end
        isempty(pair_scores) && break
        best = argmax(pair_scores)
        push!(vocab, best)
    end
    return vocab
end


"""
    wordpiece_tokenize(word, vocab; unk_token="[UNK]", max_word_len=100) → Vector{String}

Tokenize a word using greedy longest-match-first WordPiece algorithm.
Continuation tokens are prefixed with "##".
Returns `[unk_token]` if the word cannot be tokenized.
"""
function wordpiece_tokenize(word::String, vocab::Set{String}; unk_token::String="[UNK]", max_word_len::Int=100)::Vector{String}
    if length(word) > max_word_len
        return [unk_token]
    end
    tokens = String[]
    start = 1
    while start <= length(word)
        found = false
        for stop in length(word):-1:start
            substr = word[start:stop]
            candidate = start == 1 ? substr : "##" * substr
            if candidate in vocab
                push!(tokens, candidate)
                start = stop + 1
                found = true
                break
            end
        end
        if !found
            return [unk_token]
        end
    end
    return tokens
end


"""
    count_word_frequencies_streaming(filepath) → Dict{String,Int}

Count word frequencies from a file line by line without loading the entire file into memory.
"""
function count_word_frequencies_streaming(filepath::String)::Dict{String,Int}
    if !isfile(filepath)
        error("corpus file not found: $filepath")
    end
    frequencies = Dict{String,Int}()
    open(filepath, "r") do io
        for line in eachline(io)
            for word in split(strip(line))
                w = String(word)
                frequencies[w] = get(frequencies, w, 0) + 1
            end
        end
    end
    return frequencies
end


"""
    train_bpe_streaming(filepath, num_merges; verbose=false, min_frequency=0)

Train BPE by streaming word frequencies from a file instead of loading the full corpus.
Uses `count_word_frequencies_streaming` for memory-efficient frequency counting,
then runs the standard merge loop.
"""
function train_bpe_streaming(
    filepath::String,
    num_merges::Int;
    verbose::Bool=false,
    min_frequency::Int=0
)::Tuple{Dict{Vector{String},Int},Vector{Tuple{String,String}}}
    frequencies = count_word_frequencies_streaming(filepath)
    word_symbols = initialize_word_symbols(frequencies)
    merges = Tuple{String,String}[]

    for i in 1:num_merges
        pair_counts = count_pairs(word_symbols)
        pair = best_pair(pair_counts)
        if pair === nothing
            verbose && println("stopping early: no more pairs at step $i")
            break
        end
        if min_frequency > 0 && pair_counts[pair] < min_frequency
            verbose && println("stopping early: best pair frequency $(pair_counts[pair]) < min_frequency $min_frequency at step $i")
            break
        end
        if verbose
            println("merge $i: $(pair[1]) + $(pair[2]) -> $(pair[1])$(pair[2]) (freq=$(pair_counts[pair]))")
        end
        push!(merges, pair)
        new_word_symbols = Dict{Vector{String},Int}()
        for (symbols, freq) in word_symbols
            new_word_symbols[merge_symbols(symbols, pair)] = freq
        end
        word_symbols = new_word_symbols
    end
    return (word_symbols, merges)
end


"""
    encode_with_protected_tokens(text, merges; protected=String[])

Encode text while preserving protected token strings intact.
Protected strings are matched literally and emitted as single tokens.
"""
function encode_with_protected_tokens(
    text::String,
    merges::Vector{Tuple{String,String}};
    protected::Vector{String}=String[]
)::Vector{String}
    if isempty(protected)
        return encode_text(text, merges)
    end
    # split text around protected tokens, keeping them as-is
    parts = [text]
    for p in protected
        new_parts = String[]
        for part in parts
            chunks = split(part, p)
            for (i, chunk) in enumerate(chunks)
                if !isempty(String(chunk))
                    push!(new_parts, String(chunk))
                end
                if i < length(chunks)
                    push!(new_parts, p)
                end
            end
        end
        parts = new_parts
    end
    tokens = String[]
    for part in parts
        if part in protected
            push!(tokens, part)
        else
            append!(tokens, encode_text(part, merges))
        end
    end
    return tokens
end


"""
    validate_merges(merges) → Vector{String}

Check merge rules for consistency issues. Returns a list of warning messages.
Empty list means all merges are valid.
"""
function validate_merges(merges::Vector{Tuple{String,String}})::Vector{String}
    warnings = String[]
    seen = Set{Tuple{String,String}}()
    for (i, merge) in enumerate(merges)
        if merge in seen
            push!(warnings, "duplicate merge at position $i: $(merge[1]) + $(merge[2])")
        end
        push!(seen, merge)
        if isempty(merge[1]) || isempty(merge[2])
            push!(warnings, "empty component in merge at position $i")
        end
    end
    return warnings
end


"""
    validate_vocab_index(index) → Vector{String}

Check a vocabulary index for integrity issues: duplicate IDs, gaps, or missing entries.
Returns a list of warning messages.
"""
function validate_vocab_index(index::Dict{String,Int})::Vector{String}
    warnings = String[]
    if isempty(index)
        push!(warnings, "vocabulary index is empty")
        return warnings
    end
    ids = collect(values(index))
    if length(ids) != length(Set(ids))
        push!(warnings, "duplicate IDs found in vocabulary index")
    end
    min_id, max_id = extrema(ids)
    expected = max_id - min_id + 1
    if length(ids) != expected
        push!(warnings, "gaps in ID sequence: $(length(ids)) tokens but ID range is $min_id:$max_id")
    end
    return warnings
end


"""
    validate_tokenizer(t::BPETokenizer) → Vector{String}

Run all validation checks on a tokenizer. Returns a list of warning messages.
"""
function validate_tokenizer(t::BPETokenizer)::Vector{String}
    warnings = String[]
    append!(warnings, validate_merges(t.merges))
    append!(warnings, validate_vocab_index(t.vocab_index))
    # check that special tokens are in the index
    for token in t.special_tokens
        if !haskey(t.vocab_index, token)
            push!(warnings, "special token '$token' missing from vocabulary index")
        end
    end
    # check id_to_token matches vocab_index
    if length(t.id_to_token) != length(t.vocab_index)
        push!(warnings, "id_to_token size ($(length(t.id_to_token))) != vocab_index size ($(length(t.vocab_index)))")
    end
    return warnings
end

end

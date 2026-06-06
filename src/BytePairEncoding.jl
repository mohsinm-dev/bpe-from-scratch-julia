module BytePairEncoding

export word_to_symbols,
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
    most_common_tokens


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
    return BPETokenizer(merges, vocab, vocab_index, id_to_token, special_tokens)
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

end

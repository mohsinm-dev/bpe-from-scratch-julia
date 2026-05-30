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
    vocab_size_history


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
    train_bpe(corpus, num_merges)

Run the full BPE training loop for a given number of merges.

Returns the final vocabulary (word_symbols) and the list of merges performed.
"""
function train_bpe(corpus::String, num_merges::Int)::Tuple{Dict{Vector{String},Int},Vector{Tuple{String,String}}}
    frequencies = count_word_frequencies(corpus)
    word_symbols = initialize_word_symbols(frequencies)
    merges = Tuple{String,String}[]

    for i in 1:num_merges
        pair_counts = count_pairs(word_symbols)
        pair = best_pair(pair_counts)

        if pair === nothing
            break
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

end

module BytePairEncoding

export word_to_symbols,
    count_word_frequencies,
    initialize_word_symbols,
    count_pairs,
    best_pair,
    merge_symbols,
    train_bpe,
    get_vocabulary


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

end

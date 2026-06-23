#!/usr/bin/env julia

# Convert between tokenizer formats
#
# Usage:
#   julia scripts/convert.jl <input_file> <output_file> <format>
#
# Formats: hf (HuggingFace), tsv (internal), sp (SentencePiece vocab)
#
# Examples:
#   julia scripts/convert.jl merges.tsv merges.txt hf
#   julia scripts/convert.jl merges.txt merges.tsv tsv

include(joinpath(@__DIR__, "..", "src", "BytePairEncoding.jl"))
using .BytePairEncoding

function main()
    if length(ARGS) < 3
        println(stderr, "Usage: julia scripts/convert.jl <input_file> <output_file> <format>")
        println(stderr, "Formats: hf, tsv")
        exit(1)
    end

    input_file = ARGS[1]
    output_file = ARGS[2]
    format = ARGS[3]

    if format == "hf"
        merges = load_merges(input_file)
        export_huggingface_merges(merges, output_file)
        println("Converted $(length(merges)) merges to HuggingFace format: $output_file")
    elseif format == "tsv"
        merges = import_huggingface_merges(input_file)
        save_merges(merges, output_file)
        println("Converted $(length(merges)) merges to TSV format: $output_file")
    else
        println(stderr, "Unknown format: $format. Use 'hf' or 'tsv'.")
        exit(1)
    end
end

main()

.PHONY: test bench examples clean

test:
	julia test/runtests.jl

bench:
	julia benchmarks/training.jl
	julia benchmarks/encoding.jl
	julia benchmarks/byte_level.jl
	julia benchmarks/memory.jl
	julia benchmarks/cache.jl

bench-parallel:
	julia --threads=4 benchmarks/parallel.jl

examples:
	julia examples/basic_training.jl
	julia examples/byte_level.jl
	julia examples/custom_tokenizer.jl
	julia examples/multilingual.jl
	julia examples/streaming_encode.jl
	julia examples/training_comparison.jl

clean:
	rm -rf /tmp/bpe_out output/

analyze:
	julia scripts/analyze.jl /tmp/bpe_out "the quick brown fox"

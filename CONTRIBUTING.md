# Contributing

Contributions are welcome. Here's how to get started.

## Setup

```bash
git clone https://github.com/your-username/bpe-from-scratch-julia.git
cd bpe-from-scratch-julia
julia test/runtests.jl
```

Requires Julia 1.9 or later.

## Guidelines

- Keep changes focused. One concern per PR.
- Add tests for new features in `test/runtests.jl`.
- Run the full test suite before submitting.
- Follow existing code style: 4-space indentation, docstrings on public functions.
- No external dependencies unless absolutely necessary — this project is pure Julia by design.

## Adding a new tokenization algorithm

1. Implement the training function and tokenize function in `src/BytePairEncoding.jl`.
2. Add the export to the module's export list.
3. Add tests in `test/runtests.jl` under a descriptive `@testset`.
4. Update the README with usage examples.
5. Update `CHANGELOG.md`.

## Benchmarks

If your change affects performance, run the relevant benchmarks before and after:

```bash
julia benchmarks/training.jl
julia benchmarks/encoding.jl
```

Or run all benchmarks at once: `make bench`

Include benchmark results in your PR description if you claim a performance improvement.

## Testing

Run the full test suite:

```bash
julia test/runtests.jl
```

Smoke-test examples to make sure they still run:

```bash
make examples
```

## Reporting issues

Open an issue with:
- What you expected to happen
- What actually happened
- Minimal reproduction steps
- Julia version (`julia --version`)

## Adding new functions

When adding a new exported function:

1. Add the export statement in the appropriate section of `BytePairEncoding.jl`
2. Include a docstring with signature, description, and example if non-obvious
3. Add tests in `test/runtests.jl` covering normal use and edge cases
4. Update the API reference in `README.md` if the function is user-facing

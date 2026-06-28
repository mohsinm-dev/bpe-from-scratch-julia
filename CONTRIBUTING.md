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

## Reporting issues

Open an issue with:
- What you expected to happen
- What actually happened
- Minimal reproduction steps
- Julia version (`julia --version`)

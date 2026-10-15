from pathlib import Path


def validate_torch_location(torch_file):
    """PyTorch's extension linker can emit unquoted -L paths."""
    library = str(Path(torch_file).resolve().parent / 'lib')
    if any(c.isspace() for c in library):
        raise RuntimeError(
            f'PyTorch library path contains whitespace: {library}. '
            'Create a fresh environment at ~/.venvs/world-gs with bash setup.sh, '
            'then use ./run.sh train. The project and generated data can stay in world GS.'
        )

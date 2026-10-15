"""Gaussian PLY I/O, including Unity-compatible little-endian output."""
from pathlib import Path
import argparse
import os
import tempfile
import numpy as np

BASE_FIELDS = ['x', 'y', 'z', 'nx', 'ny', 'nz', 'f_dc_0', 'f_dc_1', 'f_dc_2',
               'opacity', 'scale_0', 'scale_1', 'scale_2', 'rot_0', 'rot_1', 'rot_2', 'rot_3']
FULL_FIELDS = BASE_FIELDS[:9] + [f'f_rest_{i}' for i in range(45)] + BASE_FIELDS[9:]
REQUIRED = set(BASE_FIELDS) - {'nx', 'ny', 'nz'}


def read_gaussian_ply(path):
    """Read scalar float32 ASCII/binary LE Gaussian PLYs from this pipeline."""
    with open(path, 'rb') as stream:
        if stream.readline().strip() != b'ply':
            raise ValueError('Not a PLY file')
        fields, count, encoding = [], None, None
        for _ in range(4096):
            line = stream.readline(4096)
            if not line:
                raise ValueError('Incomplete PLY header')
            tokens = line.decode('ascii').strip().split()
            if not tokens or tokens[0] in ('comment', 'obj_info'):
                continue
            if tokens == ['end_header']:
                break
            if tokens[0] == 'format':
                if len(tokens) != 3 or tokens[2] != '1.0':
                    raise ValueError('Unsupported PLY version')
                encoding = tokens[1]
            elif tokens[0] == 'element':
                if len(tokens) != 3 or tokens[1] != 'vertex' or count is not None:
                    raise ValueError('Expected only one vertex element')
                count = int(tokens[2])
            elif tokens[0] == 'property':
                if count is None or len(tokens) != 3 or tokens[1] not in ('float', 'float32'):
                    raise ValueError('Expected scalar float32 vertex properties')
                fields.append(tokens[2])
            else:
                raise ValueError('Unsupported header statement')
        else:
            raise ValueError('PLY header too long')
        if count is None or count <= 0 or len(set(fields)) != len(fields) or not REQUIRED <= set(fields):
            raise ValueError('Missing or invalid Gaussian vertex properties')
        if encoding == 'ascii':
            values = np.loadtxt(stream, dtype=np.float32, ndmin=2)
        elif encoding == 'binary_little_endian':
            expected = count*len(fields)*4
            remaining = os.fstat(stream.fileno()).st_size - stream.tell()
            if remaining != expected:
                raise ValueError(f'PLY payload size mismatch: expected {expected}, got {remaining}')
            values = np.frombuffer(stream.read(), dtype='<f4').reshape(count, len(fields))
        else:
            raise ValueError('Expected ASCII or binary_little_endian PLY')
        if values.shape != (count, len(fields)) or not np.isfinite(values).all():
            raise ValueError('Invalid vertex count, shape, or non-finite values')
        return fields, values


def write_gaussian_ply(path, values, fields=BASE_FIELDS):
    """Store logits/log scales unchanged, padding missing SH1-3 with zeros."""
    values = np.asarray(values)
    if (values.ndim != 2 or len(values) == 0 or values.shape[1] != len(fields)
            or len(set(fields)) != len(fields) or not REQUIRED <= set(fields)):
        raise ValueError('Invalid Gaussian properties or array shape')
    if set(fields) - set(FULL_FIELDS):
        raise ValueError('Unsupported Gaussian properties; only SH0 through SH3 are supported')
    data = np.zeros((len(values), len(FULL_FIELDS)), dtype='<f4')
    for index, name in enumerate(fields):
        data[:, FULL_FIELDS.index(name)] = values[:, index]
    if not np.isfinite(data).all():
        raise ValueError('Gaussian values must be finite float32')
    header = ('ply\nformat binary_little_endian 1.0\nelement vertex '+str(len(data))+'\n'
              + ''.join(f'property float {name}\n' for name in FULL_FIELDS) + 'end_header\n')
    path = Path(path)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, suffix='.tmp', delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(header.encode('ascii'))
            stream.write(data.tobytes(order='C'))
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def convert(source, output):
    source, output = Path(source), Path(output)
    if source.resolve() == output.resolve() or output.exists():
        raise FileExistsError('Choose a new output path; source and existing files are preserved')
    fields, data = read_gaussian_ply(source)
    write_gaussian_ply(output, data, fields)
    print(f'Converted {len(data)} Gaussians to binary_little_endian: {output}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    convert(args.source, args.output)


if __name__ == '__main__':
    main()

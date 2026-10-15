from pathlib import Path
import tempfile
import unittest
import numpy as np
from worldgs.ply import BASE_FIELDS, FULL_FIELDS, convert, read_gaussian_ply, write_gaussian_ply


class PlyTest(unittest.TestCase):
    def test_binary_layout_and_roundtrip(self):
        values = (np.arange(34, dtype=np.float32).reshape(2, 17)+1.25)/10
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'scene.ply'
            write_gaussian_ply(path, values)
            header, payload = path.read_bytes().split(b'end_header\n', 1)
            self.assertIn(b'format binary_little_endian 1.0', header)
            self.assertEqual(len(payload), 2*62*4)
            self.assertEqual(payload[:4], np.array([values[0, 0]], dtype='<f4').tobytes())
            fields, result = read_gaussian_ply(path)
            self.assertEqual(fields, FULL_FIELDS)
            for i, name in enumerate(BASE_FIELDS):
                np.testing.assert_array_equal(result[:, fields.index(name)], values[:, i])
            np.testing.assert_array_equal(result[:, 9:54], 0)

    def test_ascii_conversion_preserves_parameters_and_source(self):
        with tempfile.TemporaryDirectory() as folder:
            source, target = Path(folder)/'ascii.ply', Path(folder)/'unity.ply'
            values = np.arange(17, dtype=np.float32)[None]/7
            header = 'ply\nformat ascii 1.0\nelement vertex 1\n'+''.join(f'property float {f}\n' for f in BASE_FIELDS)+'end_header\n'
            source.write_text(header+' '.join(map(str, values[0]))+'\n')
            original = source.read_bytes()
            convert(source, target)
            fields, result = read_gaussian_ply(target)
            for i, name in enumerate(BASE_FIELDS):
                np.testing.assert_allclose(result[:, fields.index(name)], values[:, i])
            self.assertEqual(source.read_bytes(), original)
            with self.assertRaises(FileExistsError):
                convert(source, target)

    def test_reject_truncated_binary(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'bad.ply'
            write_gaussian_ply(path, np.zeros((1, 17)))
            path.write_bytes(path.read_bytes()[:-1])
            with self.assertRaisesRegex(ValueError, 'payload size'):
                read_gaussian_ply(path)

    def test_reject_nonfinite_before_replacing_output(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'scene.ply'
            path.write_bytes(b'preserve')
            values = np.zeros((1, 17))
            values[0, 0] = np.nan
            with self.assertRaises(ValueError):
                write_gaussian_ply(path, values)
            self.assertEqual(path.read_bytes(), b'preserve')

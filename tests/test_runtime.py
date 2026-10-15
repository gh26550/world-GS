import unittest
from worldgs.runtime import validate_torch_location


class RuntimeTest(unittest.TestCase):
    def test_reject_space_in_torch_library_path(self):
        with self.assertRaisesRegex(RuntimeError, 'whitespace'):
            validate_torch_location('/home/user/world GS/.venv/site-packages/torch/__init__.py')

    def test_allow_external_environment(self):
        validate_torch_location('/home/user/.venvs/world-gs/site-packages/torch/__init__.py')

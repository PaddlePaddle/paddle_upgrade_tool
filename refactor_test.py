import os
import unittest
import textwrap
from bowler import Query
from tempfile import NamedTemporaryFile

from refactor import *

def _refactor_helper(refactor_func, input_src, change_spec) -> str:
    try:
        ntf = NamedTemporaryFile(suffix='.py', delete=False)
        ntf.write(input_src.encode('utf-8'))
        ntf.close()
        q = Query(ntf.name)
        refactor_func(q, change_spec=change_spec).execute(interactive=False, write=True, silent=True)
        with open(ntf.name, 'r') as f:
            output_src = f.read()
        return output_src
    finally:
        os.remove(ntf.name)


class TestRefactorDemo(unittest.TestCase):
    change_spec = {}

    def test_rename_1(self):
        input_src = '''
        old_api()
        '''
        expected_src = '''
        new_api()
        '''
        input_src = textwrap.dedent(input_src).strip() + '\n'
        expected_src = textwrap.dedent(expected_src).strip() + '\n'
        output_src = _refactor_helper(refactor_demo, input_src, self.change_spec)
        self.assertEqual(output_src, expected_src)


if __name__ == '__main__':
    unittest.main()

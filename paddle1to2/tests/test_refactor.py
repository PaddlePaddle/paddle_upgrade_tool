import os
import sys
import unittest
import textwrap
from tempfile import NamedTemporaryFile

from bowler import Query
from paddle1to2.refactor import * 

def _refactor_helper(refactor_func, input_src, change_spec) -> str:
    try:
        ntf = NamedTemporaryFile(suffix='.py', delete=False)
        ntf.write(input_src.encode('utf-8'))
        ntf.close()
        q = Query(ntf.name)
        refactor_func(q, change_spec).execute(interactive=False, write=True, silent=True)
        with open(ntf.name, 'r') as f:
            output_src = f.read()
        return output_src
    finally:
        os.remove(ntf.name)


class TestRefactorDemo(unittest.TestCase):
    change_spec = {}

    def test_refactor_demo(self):
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


class TestRefactorImport(unittest.TestCase):
    def _run(self, change_spec, input_src, expected_src):
        input_src = textwrap.dedent(input_src).strip() + '\n'
        expected_src = textwrap.dedent(expected_src).strip() + '\n'
        output_src = _refactor_helper(refactor_import, input_src, change_spec)
        self.assertEqual(output_src, expected_src)

    def test_refactor_import(self):
        input_src = '''
        import paddle
        '''
        expected_src = '''
        import paddle
        '''
        self._run({}, input_src, expected_src)
        #--------------
        input_src = '''
        import paddle.fluid as fluid
        '''
        expected_src = '''
        import paddle
        '''
        self._run({}, input_src, expected_src)
        #--------------
        input_src = '''
        import paddle
        import paddle.fluid as fluid
        '''
        expected_src = '''
        import paddle
        '''
        self._run({}, input_src, expected_src)
        #--------------
        input_src = '''
        import paddle
        import paddle.fluid as fluid
        '''
        expected_src = '''
        import paddle
        '''
        self._run({}, input_src, expected_src)
        #--------------
        input_src = '''
        import paddle
        import paddle.fluid as fluid
        fluid.api()

        def func():
            fluid.api()
        '''
        expected_src = '''
        import paddle
        paddle.fluid.api()

        def func():
            paddle.fluid.api()
        '''
        self._run({}, input_src, expected_src)
        #--------------
        input_src = '''
        import paddle.fluid as fluid
        fluid.api()

        def func():
            fluid.api()
        '''
        expected_src = '''
        import paddle
        paddle.fluid.api()

        def func():
            paddle.fluid.api()
        '''
        self._run({}, input_src, expected_src)
        #--------------
        input_src = '''
        from paddle.fluid.layers import Layer

        class CustomLayer(Layer):
            pass
        print(Layer.__name__)
        print(type(Layer))
        '''
        expected_src = '''
        import paddle

        class CustomLayer(paddle.fluid.layers.Layer):
            pass
        print(paddle.fluid.layers.Layer.__name__)
        print(type(paddle.fluid.layers.Layer))
        '''
        self._run({}, input_src, expected_src)


class TestNormApiAlias(unittest.TestCase):
    change_spec = {
            "paddle.fluid.Layer": {
                "alias": [
                    "paddle.fluid.layers.Layer",
                    "paddle.fluid.layers1.layers2.Layer",
                    ]
                }
            }

    def _run(self, change_spec, input_src, expected_src):
        input_src = textwrap.dedent(input_src).strip() + '\n'
        expected_src = textwrap.dedent(expected_src).strip() + '\n'
        output_src = _refactor_helper(norm_api_alias, input_src, change_spec)
        self.assertEqual(output_src, expected_src)

    def test_norm_api_alias(self):
        input_src = '''
        import paddle

        layer = paddle.fluid.Layer()
        layer = paddle.fluid.layers.Layer()
        layer = paddle.fluid.layers.Layer_With_Underscore()
        layer = paddle.fluid.layers1.layers2.Layer()
        '''
        expected_src = '''
        import paddle

        layer = paddle.fluid.Layer()
        layer = paddle.fluid.Layer()
        layer = paddle.fluid.layers.Layer_With_Underscore()
        layer = paddle.fluid.Layer()
        '''
        self._run(self.change_spec, input_src, expected_src)


class TestApiRename(unittest.TestCase):
    change_spec = {
            "paddle.fluid.Layer": {
                "update_to": "paddle.Layer",
                },
            }

    def _run(self, change_spec, input_src, expected_src):
        input_src = textwrap.dedent(input_src).strip() + '\n'
        expected_src = textwrap.dedent(expected_src).strip() + '\n'
        output_src = _refactor_helper(api_rename_and_warning, input_src, change_spec)
        self.assertEqual(output_src, expected_src)

    def test_rename_and_warning(self):
        input_src = '''
        import paddle

        layer = paddle.fluid.Layer()
        layer = paddle.fluid.Layer_With_Underscore()
        '''
        expected_src = '''
        import paddle

        layer = paddle.Layer()
        layer = paddle.fluid.Layer_With_Underscore()
        '''
        self._run(self.change_spec, input_src, expected_src)
 

if __name__ == '__main__':
    unittest.main()

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
        output_src = _refactor_helper(api_rename, input_src, change_spec)
        self.assertEqual(output_src, expected_src)

    def test_rename(self):
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

class TestArgsToKwargs(unittest.TestCase):
    change_spec = {
            "paddle.add": {
                "args_list": ["x", "y"],
                },
            }
    def _run(self, change_spec, input_src, expected_src):
        input_src = textwrap.dedent(input_src).strip() + '\n'
        expected_src = textwrap.dedent(expected_src).strip() + '\n'
        output_src = _refactor_helper(args_to_kwargs, input_src, change_spec)
        self.assertEqual(output_src, expected_src)

    def test_args_to_kwargs(self):
        input_src = '''
        paddle.add(1,2)
        paddle.add(1, 2)
        paddle.add(1, y=2)
        paddle.add(1)
        paddle.add(z=1)
        paddle.add(paddle.to.api, paddle.to.api())
        '''
        expected_src = '''
        paddle.add(x=1,y=2)
        paddle.add(x=1, y=2)
        paddle.add(x=1, y=2)
        paddle.add(x=1)
        paddle.add(z=1)
        paddle.add(x=paddle.to.api, y=paddle.to.api())
        '''
        self._run(self.change_spec, input_src, expected_src)

class TestRefactorKwargs(unittest.TestCase):
    change_spec = {
        "paddle.add": {
            "args_change": [
                [ "x", "x_new" ],
                [ "out", "" ],
                [ "", "name", "test" ]
            ],
            "args_warning": {
                "x_new": "x_new is deleted in paddle.add"
            }
        }
    }

    def _run(self, change_spec, input_src, expected_src):
        input_src = textwrap.dedent(input_src).strip() + '\n'
        expected_src = textwrap.dedent(expected_src).strip() + '\n'
        output_src = _refactor_helper(refactor_kwargs, input_src, change_spec)
        self.assertEqual(output_src, expected_src)

    def test_refactor_kwargs(self):
        input_src = '''
        paddle.add(x=1, out=2)
        paddle.add(1)
        paddle.add()
        paddle.add(a=1, b=2, c=3)
        '''
        expected_src = '''
        paddle.add(x_new=1, name=test)
        paddle.add(1, name=test)
        paddle.add(name=test)
        paddle.add(a=1, b=2, c=3, name=test)
        '''
        self._run(self.change_spec, input_src, expected_src)

        input_src = '''
        # comment line1
        # comment line2
        # comment line3
        paddle.add(x=1, out=2)
        '''
        expected_src = '''
        # comment line1
        # comment line2
        # comment line3
        paddle.add(x_new=1, name=test)
        '''
        self._run(self.change_spec, input_src, expected_src)


class TestWithRefactor(unittest.TestCase):
    change_spec = {}

    def _run(self, change_spec, input_src, expected_src):
        input_src = textwrap.dedent(input_src).strip() + '\n'
        expected_src = textwrap.dedent(expected_src).strip() + '\n'
        output_src = _refactor_helper(refactor_with, input_src, change_spec)
        self.assertEqual(output_src, expected_src)

    def test_rename(self):
        input_src = '''
        import paddle

        with paddle.fluid.dygraph.guard(place):
            pass

        with paddle.fluid.dygraph.guard():
            pass
        '''
        expected_src = '''
        import paddle

        paddle.disable_static(place)
        pass

        paddle.disable_static()
        pass
        '''
        self._run(self.change_spec, input_src, expected_src)

        input_src = '''
        import paddle

        with fluid.dygraph.guard(place):
            pass
            pass

        with fluid.dygraph.guard():
            pass
            pass
        '''
        expected_src = '''
        import paddle

        paddle.disable_static(place)
        pass
        pass

        paddle.disable_static()
        pass
        pass
        '''
        self._run(self.change_spec, input_src, expected_src)

        input_src = '''
        import paddle

        with dygraph.guard(place):
            pass
            pass
            pass

        with dygraph.guard():
            pass
            pass
            pass
        '''
        expected_src = '''
        import paddle

        paddle.disable_static(place)
        pass
        pass
        pass

        paddle.disable_static()
        pass
        pass
        pass
        '''
        self._run(self.change_spec, input_src, expected_src)

        input_src = '''
        import paddle

        # comment line1
        with dygraph.guard(place):
            pass
            pass

        # comment line2
        # comment line3
        with dygraph.guard():
            pass
            pass
        '''
        expected_src = '''
        import paddle

        # comment line1
        paddle.disable_static(place)
        pass
        pass

        # comment line2
        # comment line3
        paddle.disable_static()
        pass
        pass
        '''
        self._run(self.change_spec, input_src, expected_src)

        input_src = '''
        import paddle

        if True is True:
            pass
            if place is None:
                pass
                with paddle.fluid.dygraph.guard():
                    pass
                    pass
                pass
            else:
                pass
        '''
        expected_src = '''
        import paddle

        if True is True:
            pass
            if place is None:
                pass
                paddle.disable_static()
                pass
                pass
                pass
            else:
                pass
        '''
        self._run(self.change_spec, input_src, expected_src)



if __name__ == '__main__':
    unittest.main()

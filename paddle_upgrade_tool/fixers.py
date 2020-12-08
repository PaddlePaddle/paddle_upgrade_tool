import logging

from fissix import fixer_base
from fissix.refactor import RefactoringTool

__all__ = [
        'FixerDemo',
        ]

class FixerDemo(fixer_base.BaseFix):
    BM_compatible = True
    # match all function call
    PATTERN = """power< any* >"""

    def __init__(self):
        _logger = logging.getLogger("RefactoringTool")
        super(FixerDemo, self).__init__(RefactoringTool._default_options, _logger)

    def transform(self, node, results):
        print('code passed to transform:', node)
        print('results passed to transform:', results)
        return node



from bowler.types import LN, Capture, Filename

__all__ = [
        'print_match',
        ]

def print_match(node: LN, capture: Capture, filename: Filename) -> bool:
    print('filename:', filename)
    print('code:\n"""{}"""\n'.format(str(node)))
    print('capture:\n"""{}"""\n'.format(capture))
    print('-' * 10)
    return True

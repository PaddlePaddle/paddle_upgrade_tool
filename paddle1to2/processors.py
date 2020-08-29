__all__ = [
        'demo_post_processor',
        ]

def demo_post_processor(filename, hunks):
    print('filename from processor:', filename)
    print('hunks from processor:', hunks)
    return True

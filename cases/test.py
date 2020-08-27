"""
this is docstring
"""
from path.to import old_api

# this is comment 1
a = old_api(1, 2)
# this is comment 2
b = path.to.old_api(1, 2)
b = path.to.old_api(args=1, 2)

c = path.to.old_api_alias1(1, 2)
d = path.to1.to2.old_api_alias2(1, 2)

class CClass:
    pass

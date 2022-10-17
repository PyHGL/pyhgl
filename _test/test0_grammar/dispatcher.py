
from pyhgl.array import *
from typing import Any

test = Dispatcher()
test.dispatch('Add', lambda x,y: print('Add'), [str, int], Any)
test.call('Add',1, 1.2)
test.dispatch('Add', lambda x,y: print('Add2'), [float, str, int], [float])
test.call('Add',1.1, 1.2)
test.dispatch('make', lambda x: print('make'), [str, int])
test2 = test.copy
test.dispatch('make', lambda x: x*2, [str, int])
test2.dispatch('Add2', lambda x: print(x), Any)
test2.call('Add2', 'xxx')
print(test)
print(test2)

import time 
t = time.time()
for i in range(100000):
    test.call('make', 'xxx') 
print('cost: ', time.time()-t)
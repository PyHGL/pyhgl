
from pyhgl.array import *
from pyhgl.tester import *
from typing import Any


res = []

@tester
def test_dispatch(self):
    test = Dispatcher()
    test.dispatch('Add', lambda x,y: res.append('Add'), [str, int], Any)
    test.call('Add',1, 1.2)
    test.dispatch('Add', lambda x,y: res.append('Add2'), [float, str, int], [float])
    test.call('Add',1.1, 1.2)
    test.dispatch('make', lambda x: x*2, [str, int])
    test2 = test.copy
    test.dispatch('make', lambda x: res.append(x), [str, int])
    test2.dispatch('Add2', lambda x: res.append(x), Any)
    test2.call('Add2', 'xxx')
    
    self.AssertEq(res, ['Add', 'Add2', 'xxx']) 

    import time 
    t = time.time()
    for i in range(10000):
        test.call('make', 'xxx') 
    self.AssertEq(len(res), 10003)
    
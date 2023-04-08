from pyhgl.array import *
from pyhgl.tester import tester 

import numpy as np


@tester 
def test_array():
    a = Array([
        [],
        [{"x": 1.1, "y": [1]}, {"x": 2.2, "y": [1, 2]}, {"x": 3.3, "y": [1, 2, 3]}],
        [{"x": 4.4, "y": [1, 2, 3, 4], 'j': 3}, {"x": 5.5, "y": [1, 2, 3, 4, 5]}]
    ], recursive=True, atoms=[str]) 

    tester.EQ += str(a), 'Vec{\n  Vec{}\n  Vec{\n    Bundle{\n      x: 1.1\n      y: Vec{\n        1\n      }\n    }\n    Bundle{\n      x: 2.2\n      y: Vec{\n        1\n        2\n      }\n    }\n    Bundle{\n      x: 3.3\n      y: Vec{\n        1\n        2\n        3\n      }\n    }\n  }\n  Vec{\n    Bundle{\n      x: 4.4\n      y: Vec{\n        1\n        2\n        3\n        4\n      }\n      j: 3\n    }\n    Bundle{\n      x: 5.5\n      y: Vec{\n        1\n        2\n        3\n        4\n        5\n      }\n    }\n  }\n}'

    tester.EQ += str(a[1:,:,'y',1:]), 'Vec{\n  Vec{\n    Vec{}\n    Vec{\n      2\n    }\n    Vec{\n      2\n      3\n    }\n  }\n  Vec{\n    Vec{\n      2\n      3\n      4\n    }\n    Vec{\n      2\n      3\n      4\n      5\n    }\n  }\n}'

    b = Bundle(
        x = 1,
        y = 2,
        z = Bundle(
            m = Bundle('a',2.0,'m0'),
            n = Bundle(-1, 'n0','n1'),
        ),
        zz = Bundle(
            m = Bundle('a',2.0,'m1'),
            n = Bundle(2, 'n2','n3'),
        )
    )   
    tester.EQ += str(b[['z','zz'], :, -1]), 'Bundle{\n  z: Bundle{\n    m: m0\n    n: n1\n  }\n  zz: Bundle{\n    m: m1\n    n: n3\n  }\n}'
    tester.EQ += str(b[{'sel1':'z', 'sel2':'zz'}, None, 'm', ::-1]), 'Bundle{\n  sel1: Vec{\n    Vec{\n      m0\n      2.0\n      a\n    }\n  }\n  sel2: Vec{\n    Vec{\n      m1\n      2.0\n      a\n    }\n  }\n}'

    b[['z','zz'], :, -1] = Array([['c','d'], ['e']]) 
    tester.EQ += str(b), "Bundle{\n  x: 1\n  y: 2\n  z: Bundle{\n    m: Vec{\n      a\n      2.0\n      ['c', 'd']\n    }\n    n: Vec{\n      -1\n      n0\n      ['c', 'd']\n    }\n  }\n  zz: Bundle{\n    m: Vec{\n      a\n      2.0\n      ['e']\n    }\n    n: Vec{\n      2\n      n2\n      ['e']\n    }\n  }\n}"


    c = Array([
        [1,2,3,4],
        [5,6,7,8]
    ], recursive=True) 
    tester.EQ += c._shape, (2, 4)
    tester.EQ += str(c[1:,1:]), 'Vec{\n  Vec{\n    6\n    7\n    8\n  }\n}'

    tester.EQ += str(Map(lambda x,y: x+y, b, b)), "Bundle{\n  x: 2\n  y: 4\n  z: Bundle{\n    m: Vec{\n      aa\n      4.0\n      ['c', 'd', 'c', 'd']\n    }\n    n: Vec{\n      -2\n      n0n0\n      ['c', 'd', 'c', 'd']\n    }\n  }\n  zz: Bundle{\n    m: Vec{\n      aa\n      4.0\n      ['e', 'e']\n    }\n    n: Vec{\n      4\n      n2n2\n      ['e', 'e']\n    }\n  }\n}"

 
 
    # test arraylize 
    # -------------- 
    
    @vectorize_axis
    def mysum(*args, hint) -> int:
        if hint == 'hint':
            return sum(args)  
        else:
            return -1

    x = Array(np.ones((2,3,4)))

    tester.EQ += mysum(1,2,3, hint='hint'), 6 
    tester.EQ += list(mysum([1,2],3,[4,5], hint='hint')._flat), [8, 10] 
    tester.EQ += mysum(x, hint='hint'), 24.0
    tester.EQ += list(mysum(x, axis=-1,hint='hint')._flat), [4.0, 4.0, 4.0, 4.0, 4.0, 4.0]
    tester.EQ += list(mysum(x, axis=(1,0),hint='hint')._flat), [6.0, 6.0, 6.0, 6.0]


from pyhgl.logic import * 
from pyhgl.tester import * 

import pyhgl.logic.utils as utils
import random

sess = Session(verbose_sim=True)
sess.enter()


# cpp 
# from pyhgl.logic.module_cpp import *  

# input_v = [GlobalVar() for _ in range(5)]
# input_x = [GlobalVar() for _ in range(5)]
# nodes = [Node() for _ in range(4)]
# start = input_v[0]
# for i in range(4):
#     res = GlobalVar()
#     nodes[i].Xor(start, input_v[i+1], target=res)
#     start = res
# start = input_x[0]
# for i in range(4):
#     res = GlobalVar()
#     node = nodes[i]
#     node.dump_unknown()
#     node.Xor(start, input_x[i+1], target=res)
#     start = res

# for i in range(3):
#     nodes[i].merge(nodes[i+1])
# sess.sim_cpp.dump_graph()

@tester 
def test_logic():
    x = Logic('0011xx')
    y = Logic('1x0x10')
    
    tester.EQ += x & y, '000xx0'
    tester.EQ += x | y, '1x111x'
    tester.EQ += x ^ y, '1x1xxx'
    tester.EQ += ~x, Logic('1100xx') | Logic(~0b111111) 


@tester
def test_single_input():
    a = UInt[6](0, name='a') 
    b = UInt('11xx00')
    result_and = And(a, b, name='result_and') 
    result_not = Not(a, name='result_not') 
    
    sess.run(10)
    for _ in range(10):
        v = setx(a)
        sess.run(10)
        print(a)
        tester.EQ += getv(result_and), v & Logic('11xx00')
        tester.EQ += getv(result_not), (~v) & Logic('111111')


sess.dumpGraph()
sess.sim_cpp.dump()
sess.sim_cpp.build()
tester.summary() 
sess.dumpVerilog(top=True)

"""
@tester
def test_boolean_shift_cmp_arith():
    # signals
    inputs = Array(UInt[5](0, name=f'input_{i}') for i in range(4))
    # bitwise and/or/...
    out_and = And(inputs)
    out_nand = Nand(inputs)
    out_or = Or(inputs)
    out_nor = Nor(inputs)
    out_xor = Xor(inputs)
    out_nxor = Nxor(inputs)
    out_not = ~ inputs
    out_andr = AndR(inputs)
    out_nandr = NandR(inputs)
    out_orr = OrR(inputs) 
    out_norr = NorR(inputs)
    out_xorr = XorR(inputs)
    out_nxorr = NxorR(inputs)
    
    out_cat = Cat(inputs)
    out_duplicated = inputs ** [1,2,3,4] 
    # logic and/or/not
    out_logicand = LogicAnd(inputs)
    out_logicor = LogicOr(inputs) 
    out_logicnot = LogicNot(inputs)
    # shift
    out_lshift = inputs << [1,2,3,4]
    out_rshift = inputs >> [1,2,3,4]
    # compare
    out_eq = inputs[0] == inputs[1]
    out_ne = inputs[0] != inputs[1]
    out_lt = inputs[0] < inputs[1]
    out_le = inputs[0] <= inputs[1]
    out_gt = inputs[0] > inputs[1]
    out_ge = inputs[0] >= inputs[1]
    # arithmetic 
    out_pos = +inputs 
    out_neg = -inputs
    out_add = Add(inputs)
    out_sub = Sub(inputs)
    out_mul = Mul(inputs)
    out_floordiv = inputs[0] // inputs[1] 
    out_mod = inputs[0] % inputs[1] 
    # simple slicing 
    out_sliced = inputs[:,-3:] 
    # mux 
    out_mux = Mux(inputs[0], inputs[1], inputs[2])
    
    sess.run(10)  
    # tests
    for n in range(100):
        
        v =  Array(random.randint(0,31) for _ in range(4)) 
        setv(inputs, v)
        sess.run(10)  
        
        tester.EQ += getv(out_and), v[0] & v[1] & v[2] & v[3] 
        tester.EQ += getv(out_nand), ~(v[0] & v[1] & v[2] & v[3]) & 31 
        tester.EQ += getv(out_or), v[0] | v[1] | v[2] | v[3] 
        tester.EQ += getv(out_nor), ~(v[0] | v[1] | v[2] | v[3]) & 31 
        tester.EQ += getv(out_xor), v[0] ^ v[1] ^ v[2] ^ v[3]
        tester.EQ += getv(out_nxor), ~(v[0] ^ v[1] ^ v[2] ^ v[3]) & 31 
        
        tester.EQ += list(getv(out_not)), [~i & 31 for i in v]
        tester.EQ += list(getv(out_andr)), [i==31 for i in v] 
        tester.EQ += list(getv(out_nandr)), [not i==31 for i in v] 
        tester.EQ += list(getv(out_orr)), [not i==0 for i in v] 
        tester.EQ += list(getv(out_norr)), [i==0 for i in v] 
        tester.EQ += list(getv(out_xorr)), [utils.parity(i) for i in v] 
        tester.EQ += list(getv(out_nxorr)), [not utils.parity(i) for i in v] 
        
        tester.EQ += getv(out_cat), v[0] | v[1] << 5 | v[2] << 10 | v[3] << 15  
        tester.EQ += list(getv(out_duplicated)), [
            v[0],
            v[1] << 5 | v[1],
            v[2] << 10 | v[2] << 5 | v[2],
            v[3] << 15 | v[3] << 10 | v[3] << 5 | v[3],
        ]
        tester.EQ += getv(out_logicand), all(v)
        tester.EQ += getv(out_logicor), any(v)
        tester.EQ += getv(out_logicnot), [not v[0], not v[1], not v[2], not v[3]]
        
        tester.EQ += list(getv(out_lshift)), [
            v[0] << 1 & 31,
            v[1] << 2 & 31,
            v[2] << 3 & 31,
            v[3] << 4 & 31,
        ]
        tester.EQ += list(getv(out_rshift)), [
            v[0] >> 1 & 31,
            v[1] >> 2 & 31,
            v[2] >> 3 & 31,
            v[3] >> 4 & 31,
        ]
        
        tester.EQ += getv(out_eq), v[0] == v[1] 
        tester.EQ += getv(out_ne), v[0] != v[1] 
        tester.EQ += getv(out_lt), v[0] < v[1] 
        tester.EQ += getv(out_le), v[0] <= v[1] 
        tester.EQ += getv(out_gt), v[0] > v[1] 
        tester.EQ += getv(out_ge), v[0] >= v[1] 
        
        tester.EQ += list(getv(out_pos)), list(v)
        tester.EQ += list(getv(out_neg)), [
            -v[0] & 31,
            -v[1] & 31, 
            -v[2] & 31,
            -v[3] & 31
        ]
        tester.EQ += getv(out_add), v[0] + v[1] + v[2] + v[3] & 31 
        tester.EQ += getv(out_sub), v[0] - v[1] - v[2] - v[3] & 31 
        tester.EQ += getv(out_mul), v[0] * v[1] * v[2] * v[3] & 31 
        if v[1] != 0:
            tester.EQ += getv(out_floordiv), v[0] // v[1]
            tester.EQ += getv(out_mod), v[0] % v[1]

        tester.EQ += list(getv(out_sliced)), [
            v[0] >> 2 & 7,
            v[1] >> 2 & 7,
            v[2] >> 2 & 7,
            v[3] >> 2 & 7,
        ]
        
        tester.EQ += getv(out_mux), v[1] if v[0] else v[2]


sess.exit()

"""

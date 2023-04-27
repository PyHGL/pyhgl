from pyhgl.logic import * 
from pyhgl.tester import * 

import pyhgl.logic.utils as utils
import random




@tester 
def test_logic(self):
    x = Logic('0011xx')
    y = Logic('1x0x10')
    
    self.AssertEq(x & y, '000xx0')
    self.AssertEq(x | y, '1x111x')
    self.AssertEq(x ^ y, '1x1xxx')
    self.AssertEq(~x, Logic('1100xx') | Logic(~0b111111) )

    z = Logic('01010x0x')
    self.AssertEq(z.to_bin(16), '0000000001010x0x')
    self.AssertEq(z.to_hex(16), '005x')
    z = Logic(-3)
    self.AssertEq(z.to_hex(15), '7ffd')
    self.AssertEq(z.to_int()  , -3)
    self.AssertEq(z.to_int(3) , -3)
    self.AssertEq(z.to_int(2) , 1 )
    self.AssertEq(list(z.split(8,4)), [0xd, 0xf, 0xf,0xf,0xf,0xf,0xf,0xf,])


@tester
def test_single_input(self):
    with Session() as sess:
        a = UInt(0, w=6, name='a') 
        b = UInt('11xx00', name='b')
        result_not = Not(a, name='result_not') 
        result_and: Reader = And(b, name='result_and') 
        sess.run(10)
        for _ in range(10):
            a_in = setx(a)
            b_in = setx(b)
            sess.run(10)
            self.EQ += getv(result_not), (~a_in) & Logic('111111')
            self.EQ += getv(result_and), b_in


sess = Session()
sess.enter()

@tester
def test_boolean_shift_cmp_arith(self):
    # 4 input signals
    inputs = Array(UInt[5](0, name=f'input_{i}') for i in range(4))
    # bitwise and/or/xor...
    out_and = And(inputs, axis=None)
    out_nand = Nand(inputs, axis=None)
    out_or = Or(inputs, axis=None)
    out_nor = Nor(inputs, axis=None)
    out_xor = Xor(inputs, axis=None)
    out_nxor = Nxor(inputs, axis=None)
    out_not = ~ inputs
    out_andr = AndR(inputs)
    out_orr = OrR(inputs) 
    out_xorr = XorR(inputs)
    
    out_cat = Cat(inputs, axis=None)
    out_duplicated = inputs ** [1,2,3,4] 
    # logic and/or/not
    out_logicand = LogicAnd(inputs, axis=None)
    out_logicor = LogicOr(inputs, axis=None) 
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
    out_add = Add(inputs, axis=None)
    out_sub = Sub(inputs, axis=None)
    out_mul = Mul(inputs, axis=None)
    out_floordiv = inputs[0] // inputs[1] 
    out_mod = inputs[0] % inputs[1] 
    # simple slicing 
    out_sliced = inputs[:,-3:] 
    # mux 
    out_mux = Mux(inputs[0], inputs[1], inputs[2])
    
    sess.run(10)  
    # tests
    for _ in range(100):
        
        v = setx(inputs) 
        mask = Logic(0b11111)
        sess.run(10)  
        
        self.EQ += getv(out_and), v[0] & v[1] & v[2] & v[3] 
        self.EQ += getv(out_nand), ~(v[0] & v[1] & v[2] & v[3]) & mask 
        self.EQ += getv(out_or), v[0] | v[1] | v[2] | v[3] 
        self.EQ += getv(out_nor), ~(v[0] | v[1] | v[2] | v[3]) & mask 
        self.EQ += getv(out_xor), v[0] ^ v[1] ^ v[2] ^ v[3]
        self.EQ += getv(out_nxor), ~(v[0] ^ v[1] ^ v[2] ^ v[3]) & mask 
        
        self.EQ += list(getv(out_not)), [~i & mask for i in v]
        self.EQ += list(getv(out_andr)), [(i | ~mask)._andr() for i in v] 
        self.EQ += list(getv(out_orr)), [i._orr() for i in v] 
        self.EQ += list(getv(out_xorr)), [i._xorr() for i in v] 
        
        self.EQ += getv(out_cat), (
            v[0] 
            | v[1] << Logic(5) 
            | v[2] << Logic(10) 
            | v[3] << Logic(15) 
        )
        self.EQ += list(getv(out_duplicated)), [
            v[0],
            v[1] << Logic(5)  | v[1],
            v[2] << Logic(10) | v[2] << Logic(5)  | v[2],
            v[3] << Logic(15) | v[3] << Logic(10) | v[3] << Logic(5)  | v[3],
        ]
        self.EQ += getv(out_logicand), v[0]._orr() & v[1]._orr() & v[2]._orr() & v[3]._orr()
        self.EQ += getv(out_logicor), v[0]._orr() | v[1]._orr() | v[2]._orr() | v[3]._orr()
        self.AssertEq(list(getv(out_logicnot)), [
            ~v[0]._orr() & Logic(1), 
            ~v[1]._orr() & Logic(1),
            ~v[2]._orr() & Logic(1),
            ~v[3]._orr() & Logic(1),
        ])
        
        self.AssertEq(list(getv(out_lshift)), [
            v[0] << Logic(1) & mask,
            v[1] << Logic(2) & mask,
            v[2] << Logic(3) & mask,
            v[3] << Logic(4) & mask,
        ])
        self.AssertEq(list(getv(out_rshift)), [
            v[0] >> Logic(1),
            v[1] >> Logic(2),
            v[2] >> Logic(3),
            v[3] >> Logic(4),
        ])
        
        self.EQ += getv(out_eq), v[0]._eq(v[1])  
        self.EQ += getv(out_ne), v[0]._ne(v[1]) 
        self.EQ += getv(out_lt), v[0]._lt(v[1]) 
        self.EQ += getv(out_le), v[0]._le(v[1])
        self.EQ += getv(out_gt), v[0]._gt(v[1])
        self.EQ += getv(out_ge), v[0]._ge(v[1])
        
        self.EQ += list(getv(out_pos)), list(v)
        self.EQ += list(getv(out_neg)), [
            -v[0] & mask,
            -v[1] & mask, 
            -v[2] & mask,
            -v[3] & mask
        ]
        self.EQ += getv(out_add), (v[0] + v[1] + v[2] + v[3]) & mask 
        self.EQ += getv(out_sub), (v[0] - v[1] - v[2] - v[3]) & mask 
        self.EQ += getv(out_mul), (v[0] * v[1] * v[2] * v[3]) & mask 
        self.EQ += getv(out_floordiv), (v[0] // v[1]) & mask
        self.EQ += getv(out_mod), (v[0] % v[1]) & mask

        self.EQ += list(getv(out_sliced)), [
            v[0] >> Logic(2) & Logic(7),
            v[1] >> Logic(2) & Logic(7),
            v[2] >> Logic(2) & Logic(7),
            v[3] >> Logic(2) & Logic(7),
        ]
        
        if v[0].x:
            _out_mux = Logic(v[1].v | v[2].v, v[1].x|v[2].x|v[1].v^v[2].v)
        elif v[0].v:
            _out_mux = v[1]
        else:
            _out_mux = v[2]
        self.EQ += getv(out_mux), _out_mux


sess.exit()


import inspect 

# ==================================== test grammar ======================================= 
from pyhgl.parser import hgl_compile, ast_dump 
from pyhgl.tester import tester

r = []

# ==================================== expr ======================================= 

# priority: same as ~
def __hgl_logicnot__(a):            
    ret = f'! {a}'
    r.append(ret)
    return ret      

# priority: above |->
def __hgl_logicand__(a,b):          
    ret = f'{a} && {b}' 
    r.append(ret)
    return ret     
    

# priority: above |->
def __hgl_logicor__(a,b):          
    ret = f'{a} || {b}' 
    r.append(ret)
    return ret     
   

# priority: above |->
def __hgl_rshift__(*args):     
    if len(args) == 2: 
        a, b = args
        ret = f'{a} >>> {b}' 
    else:
        ret = f'>>> {args[0]}'
    r.append(ret) 
    return ret  

# priority: between `not/and/or` and `comparation`
def __hgl_imply__(a,b):     
    ret = f'{a} |-> {b}' 
    r.append(ret)
    return ret        

def __hgl_unit__(a,b):
    ret = f'{a}`{type(b)}'
    r.append(ret)
    return ret  

@tester
def test_expr(self):
    exec(hgl_compile("!'a' >>> 'b'*3 || 'd' >>> !'e' + 'f' >>> 'g'")) 
    self.EQ += r, [
        '! a', 
        '! a >>> bbb', 
        '! a >>> bbb || d', 
        '! e', 
        '! a >>> bbb || d >>> ! ef', 
        '! a >>> bbb || d >>> ! ef >>> g'
    ]
    r.clear() 
    exec(hgl_compile(">>> 'a' >>> ('b' || 'c') && 'd' |-> 'x'*2 >>> 2e-2`m/s` != 1 |-> (2+2j)`N*m`")) 
    self.EQ += r, [
        '>>> a', 
        'b || c', 
        '>>> a >>> b || c', 
        '>>> a >>> b || c && d', 
        "0.02`<class 'function'>", 
        'xx >>> True', 
        '>>> a >>> b || c && d |-> xx >>> True', 
        "(2+2j)`<class 'function'>", 
        ">>> a >>> b || c && d |-> xx >>> True |-> (2+2j)`<class 'function'>"
    ]

# ==================================== assignment ======================================= 

# | t_primary '[' b=slices ']' !t_lookahead '<==' c=expression {__hgl_partial_assign__}
# | expression '<==' b=expression {__hgl_assign__}
def __hgl_partial_assign__(a,b, slices=None):     r.append(f'{a} ## {slices} <== {b}')
def __hgl_connect__(a,b):                   r.append(f'{a} <=> {b}')

@tester
def test_assign(self):
    r.clear() 
    exec(hgl_compile("a,b = 'left','right'; a <== b; a[1] <== b; a['x',:-1] <== b[-1]; a <=> b;")) 
    self.EQ += r, [
        'left ## None <== right', 
        'left ## 1 <== right', 
        "left ## ('x', slice(None, -1, None)) <== t", 
        'left <=> right'
    ]

    hgl_compile("a.b[3] <== c.d >>> 4 && x[2].y.z >>> z.z(a || b && !c)>>>1>>>1")


# ==================================== decorator ======================================= 

def module(f):
    r.append(f())
    return f 

code=""" 
@module _M1:
    x = 1 
    y = 1 
@module _M2(*args: str, **kwargs):
    pass
"""

@tester 
def test_module(self):
    r.clear()
    exec(hgl_compile(code)) 
    self.EQ += r, [{'x': 1, 'y': 1}, {'args': (), 'kwargs': {}}]


# ==================================== stmt ======================================= 

count = 0
class __hgl_when__:
    def __init__(self, arg):    r.append(arg)
    def __enter__(self):        r.append('begin_when')
    def __exit__(self, *args):  r.append('end_when')
class __hgl_elsewhen__:
    def __init__(self, arg):    r.append(arg)
    def __enter__(self):        r.append('begin_elsewhen')
    def __exit__(self, *args):  r.append('end_elsewhen')
class __hgl_otherwise__:
    def __enter__(self):        r.append('begin_otherwise')
    def __exit__(self, *args):  r.append('end_otherwise')
class __hgl_switch__:
    def __init__(self, arg):    r.append(arg)
    def __enter__(self):        r.append('begin_switch')
    def __exit__(self, *args):  r.append('end_switch')
class __hgl_once__:
    def __init__(self, arg):    r.append(arg)
    def __enter__(self):        r.append('begin_once')
    def __exit__(self, *args):  r.append('end_once')
    
    
code=""" 
when 123: 
    ...
elsewhen 456: 
    ...
otherwise: 
    switch s:='state':
        once 'idle': 
            ... 
        once 's0', 's1', 's2': 
            switch ('a','b'):
                once ('s3'): 
                    when 's4': 
                        ... 
                    elsewhen 's5': 
                        ... 
                once s:
                    ...
        once *(1,2,3): 
            ...
        once ...:
            ...
"""

@tester 
def test_stmt(self):
    r.clear()
    exec(hgl_compile(code)) 
    self.EQ += r,[123, 'begin_when', 'end_when', 456, 'begin_elsewhen', 'end_elsewhen', 'begin_otherwise', 'state', 'begin_switch', ('idle',), 'begin_once', 'end_once', ('s0', 's1', 's2'), 'begin_once', ('a', 'b'), 'begin_switch', ('s3',), 'begin_once', 's4', 'begin_when', 'end_when', 's5', 'begin_elsewhen', 'end_elsewhen', 'end_once', ('state',), 'begin_once', 'end_once', 'end_switch', 'end_once', (1, 2, 3), 'begin_once', 'end_once', (Ellipsis,), 'begin_once', 'end_once', 'end_switch', 'end_otherwise']



print(ast_dump(
"""
(a, b) <== 1
"""
))



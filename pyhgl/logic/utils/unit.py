from typing import Tuple
import re  


_number = re.compile(r"""
    [-+]?  
    (?:
        (?: \d*\.\d+ )      # .1 .12 .123 etc 9.1 etc 98.1 etc
        |(?: \d+\.? )       # 1. 12. 123. etc 1 12 123 etc
    )
    (?: [Ee] [+-]? \d+ ) ?  # exponent part
""", re.VERBOSE) 


TIME = {
    's': 1.0,
    'ms': 1e-3,
    'us': 1e-6,
    'ns': 1e-9,
    'ps': 1e-12,
    'fs': 1e-15
}


def quantity(x: str) -> Tuple[float, Tuple[float, str, dict]]:
    """
    ex. '1ns' -> (1e-9, (1e-9, 's',TIME))
        '3.3 ms' -> (3.3e-6, (3.3e-3, 'ms', TIME))'
    """
    x = x.strip()
    if r := _number.match(x).span():
        value = x[r[0]:r[1]]
        unit = x[r[1]:].strip() 
        value = float(value) 
        if unit in TIME:
            return value * TIME[unit], (value, unit, TIME)
            
    raise ValueError(f'invalid quantity: {x}') 



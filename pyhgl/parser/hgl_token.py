_globals_before = None 
_globals_before = set(globals().keys())

ENDMARKER               = None
NAME                    = None
NUMBER                  = None
STRING                  = None
NEWLINE                 = None
INDENT                  = None
DEDENT                  = None

LPAR                    = '('
RPAR                    = ')'
LSQB                    = '['
RSQB                    = ']'
COLON                   = ':'
COMMA                   = ','
SEMI                    = ';'
PLUS                    = '+'
MINUS                   = '-'
STAR                    = '*'
SLASH                   = '/'
VBAR                    = '|'
AMPER                   = '&'
LESS                    = '<'
GREATER                 = '>'
EQUAL                   = '='
DOT                     = '.'
PERCENT                 = '%'
LBRACE                  = '{'
RBRACE                  = '}'
EQEQUAL                 = '=='
NOTEQUAL                = '!='
LESSEQUAL               = '<='
GREATEREQUAL            = '>='
TILDE                   = '~'
CIRCUMFLEX              = '^'
LEFTSHIFT               = '<<'
RIGHTSHIFT              = '>>'
DOUBLESTAR              = '**'
PLUSEQUAL               = '+='
MINEQUAL                = '-='
STAREQUAL               = '*='
SLASHEQUAL              = '/='
PERCENTEQUAL            = '%='
AMPEREQUAL              = '&='
VBAREQUAL               = '|='
CIRCUMFLEXEQUAL         = '^='
LEFTSHIFTEQUAL          = '<<='
RIGHTSHIFTEQUAL         = '>>='
DOUBLESTAREQUAL         = '**='
DOUBLESLASH             = '//'
DOUBLESLASHEQUAL        = '//='
AT                      = '@'
ATEQUAL                 = '@='
RARROW                  = '->'
ELLIPSIS                = '...'
COLONEQUAL              = ':='

# pyhgl operators
LOGICNOT                = '!'
LOGICAND                = '&&'
LOGICOR                 = '||'
LLEFTSHIFT              = '<<<'
RRIGHTSHIFT             = '>>>'
LESSEQEQUAL             = '<=='
LESSEQGREATER           = '<=>'
EQEQEQUAL               = '==='
VBARRARROW              = '|->'
BACKTICK                = '`'

OP                      = None  
AWAIT                   = None
ASYNC                   = None
TYPE_IGNORE             = None
TYPE_COMMENT            = None
SOFT_KEYWORD            = None
ERRORTOKEN              = None
COMMENT                 = None
NL                      = None
ENCODING                = None 

N_TOKENS                = None
# Special definitions for cooperation with parser
NT_OFFSET               = None


_globals_after = globals().copy()




# list of (token_name:str, token_value:int, _token:str|None)
_tokens = []

for _name, _token in _globals_after.items():
    if not _name in _globals_before:
        _token_value = 256 if _name == 'NT_OFFSET' else len(_tokens) 
        _tokens.append((_name, _token_value, _token))

# {0:'ENDMARKER', 1:'NAME', ...}
tok_name = {}                
# {'(': 7, ')': 8, ...}
EXACT_TOKEN_TYPES = {}          

for _name, _token_value, _token in _tokens:
    globals()[_name] = _token_value 
    tok_name[_token_value] = _name 
    if _token is not None:
        EXACT_TOKEN_TYPES[_token] = _token_value



def ISTERMINAL(x):
    return x < NT_OFFSET

def ISNONTERMINAL(x):
    return x >= NT_OFFSET

def ISEOF(x):
    return x == ENDMARKER


__all__ = ['tok_name', 'ISTERMINAL', 'ISNONTERMINAL', 'ISEOF'] + list(tok_name.values())




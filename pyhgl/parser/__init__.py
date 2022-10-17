import ast
import io

from .hgl_parser import parse_string as hgl_parse_string
from .hgl_parser import parse_file as hgl_parse_file
from .hgl_tokenize import generate_tokens


def hgl_compile(source:str, name:str = '<string>'):
    """parse and compile in exec mode
    """
    ast_tree = hgl_parse_string(source, 'exec')
    return compile(ast_tree, name, 'exec') 
    
def hgl_tokenizer(code: str):
    return generate_tokens(io.StringIO(code).readline) 

# https://bitbucket.org/takluyver/greentreesnakes/src/master/astpp.py
def ast_dump(node, annotate_fields=True, include_attributes=False, indent='  '):
    """
    Return a formatted dump of the tree in *node*.  This is mainly useful for
    debugging purposes.  The returned string will show the names and the values
    for fields.  This makes the code impossible to evaluate, so if evaluation is
    wanted *annotate_fields* must be set to False.  Attributes such as line
    numbers and column offsets are not dumped by default.  If this is wanted,
    *include_attributes* can be set to True.
    """
    if isinstance(node, str):
        node = hgl_parse_string(node)

    def _format(node, level=0):
        if isinstance(node, ast.AST):
            fields = [(a, _format(b, level)) for a, b in ast.iter_fields(node)]
            if include_attributes and node._attributes:
                fields.extend([(a, _format(getattr(node, a), level))
                               for a in node._attributes])
            return ''.join([
                node.__class__.__name__,
                '(',
                ', '.join(('%s=%s' % field for field in fields)
                           if annotate_fields else
                           (b for a, b in fields)),
                ')'])
        elif isinstance(node, list):
            lines = ['[']
            lines.extend((indent * (level + 2) + _format(x, level + 2) + ','
                         for x in node))
            if len(lines) > 1:
                lines.append(indent * (level + 1) + ']')
            else:
                lines[-1] += ']'
            return '\n'.join(lines)
        return repr(node)

    if not isinstance(node, ast.AST):
        raise TypeError('expected AST, got %r' % node.__class__.__name__)
    return _format(node) 



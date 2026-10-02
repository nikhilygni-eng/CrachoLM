"""Narrow, local arithmetic tool. Its replies must be labelled as tool output."""
import ast
from decimal import Decimal, DecimalException, localcontext
import operator
import re

NUMBER = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)"
OPERATORS = {ast.Add: operator.add, ast.Sub: operator.sub,
             ast.Mult: operator.mul, ast.Div: operator.truediv}


def display(number):
    if number == 0:
        return "0"
    result = format(number, 'f')
    return result.rstrip('0').rstrip('.') if '.' in result else result


def calculate_expression(expression):
    """Parse arithmetic without eval, function calls, names, or attribute access."""
    if len(expression) > 160 or not re.fullmatch(r"[\d.\s()+*/-]+", expression):
        raise ValueError('Unsupported expression.')
    tree = ast.parse(expression, mode='eval')
    if len(list(ast.walk(tree))) > 64:
        raise ValueError('Expression is too long.')
    def visit(node, depth=0):
        if depth > 12:
            raise ValueError('Expression is too deeply nested.')
        if isinstance(node, ast.Expression):
            return visit(node.body, depth+1)
        if isinstance(node, ast.Constant) and type(node.value) in (int,float):
            # Preserve decimal input rather than converting through binary float.
            literal = ast.get_source_segment(expression,node)
            if len(literal) > 30:
                raise ValueError('Number is too long.')
            return Decimal(literal)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op,(ast.UAdd,ast.USub)):
            result = visit(node.operand,depth+1)
            return -result if isinstance(node.op,ast.USub) else result
        if isinstance(node, ast.BinOp) and type(node.op) in OPERATORS:
            left,right=visit(node.left,depth+1),visit(node.right,depth+1)
            result=OPERATORS[type(node.op)](left,right)
            if not result.is_finite() or abs(result)>Decimal('1e50'):
                raise ValueError('Result is outside the supported range.')
            return result
        raise ValueError('Only addition, subtraction, multiplication, and division are supported.')
    with localcontext() as context:
        context.prec=28
        return visit(tree)


def try_calculation(message):
    """Return a labelled result only for a complete, recognized math request."""
    if not isinstance(message,str) or len(message)>220:
        return None
    text=message.strip().lower().rstrip('?!').strip()
    text=re.sub(r'^(?:hi|hello|hey)[,!]?\s+', '', text)
    text=re.sub(r'^please\s+', '', text)
    match=re.fullmatch(rf'which(?: number)? is (larger|greater|bigger|smaller|less)\s*[:,]?\s*({NUMBER})\s*(?:or|and|,)\s*({NUMBER})\s*[.]?',text)
    if match:
        direction,a,b=match.groups()
        a,b=Decimal(a),Decimal(b)
        if a==b:
            reply='Both numbers are equal.'
        else:
            small=direction in ('smaller','less')
            reply=display(min(a,b) if small else max(a,b))+(' is smaller.' if small else ' is larger.')
        return {'reply':reply,'source':'local_calculator'}
    expression=re.sub(r"^(?:what is|what's|calculate|compute|evaluate)\s+",'',text)
    for word,symbol in [('multiplied by','*'),('divided by','/'),('plus','+'),('minus','-'),('times','*')]:
        expression=re.sub(r'\b'+word+r'\b',symbol,expression)
    expression=expression.translate(str.maketrans({'×':'*','÷':'/','−':'-'})).strip()
    # Reject compound instructions and unknown language instead of partially matching.
    if not re.fullmatch(r'[\d.\s()+*/-]+',expression) or not re.search(r'\d\s*[)+*/-]|[+*/-]\s*\d',expression):
        return None
    if expression.endswith('.'):
        expression=expression[:-1].rstrip()
    try:
        reply=display(calculate_expression(expression))+'.'
    except (SyntaxError,ValueError,DecimalException,ZeroDivisionError) as error:
        reply='I cannot calculate that expression. Use numbers with +, -, *, /, and parentheses; division by zero is undefined.'
    return {'reply':reply,'source':'local_calculator'}

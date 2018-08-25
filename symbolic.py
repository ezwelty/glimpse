import sympy
import re

def R(viewdir):
    """
    Return a symbolic camera rotation matrix.

    Arguments:
        viewdir (sympy.Array): Camera view directions
    """
    radians = viewdir * sympy.pi / 180
    C = sympy.Array([sympy.cos(xi) for xi in radians])
    S = sympy.Array([sympy.sin(xi) for xi in radians])
    return sympy.Matrix([
        [C[0] * C[2] + S[0] * S[1] * S[2],  C[0] * S[1] * S[2] - C[2] * S[0], -C[1] * S[2]],
        [C[2] * S[0] * S[1] - C[0] * S[2],  S[0] * S[2] + C[0] * C[2] * S[1], -C[1] * C[2]],
        [C[1] * S[0]                     ,  C[0] * C[1]                     ,  S[1]       ]
    ])

def projection(xy, va, vb):
    """
    Return a symbolic projection of camera coordinates between two cameras.

    Arguments:
        xy (sympy.Matrix): Camera A homogeneous camera coordinates [x; y; 1]
        va (sympy.Array): Camera A view directions
        vb (sympy.Array): Camera B view directions
    """
    Ra = R(va)
    Rb = R(vb)
    RR = Rb * Ra.T
    RR.simplify()
    pxy = RR * xy
    return pxy / pxy[2]

def projection_jacobian_function(path=None):
    """
    Return or write the text of a function that computes the Jacobian
    for a projection of camera coordinates.

    Function returns gradients by column for parameters:
    viewdirA0, viewdirA1, viewdirA2, viewdirB0, viewdirB1, viewdirB2
    and by row for observations:
    x0, y0, x1, y1, ...

    Arguments:
        path (str): Path to file. If `None`, returns text as str.
    """
    viewdirA0, viewdirA1, viewdirA2, xyA0, xyA1 = sympy.symbols(
        ('viewdirA0', 'viewdirA1', 'viewdirA2', 'xyA0', 'xyA1'), real=True)
    viewdirB0, viewdirB1, viewdirB2, xyB0, xyB1 = sympy.symbols(
        ('viewdirB0', 'viewdirB1', 'viewdirB2', 'xyB0', 'xyB1'), real=True)
    xyA = sympy.Matrix((xyA0, xyA1, 1))
    viewdirA = sympy.Matrix((viewdirA0, viewdirA1, viewdirA2))
    viewdirB = sympy.Matrix((viewdirB0, viewdirB1, viewdirB2))
    xyAB = projection(xyA, viewdirA, viewdirB)
    params = (viewdirA0, viewdirA1, viewdirA2, viewdirB0, viewdirB1, viewdirB2)
    derivs = [item for param in params for item in xyAB.diff(param)[0:2]]
    cse = sympy.cse(derivs, optimizations='basic')
    lines = (
        [str(var) + ' = ' + str(expr) for var, expr in cse[0]] +
        [str(cse[1])])
    replacements = [
        (r'\s*([\*\/\-])\s*', ' \\1 '),
        (r'\s*\*\s*\*\s*', '**'),
        (r' sin', ' math.sin'),
        (r' cos', ' math.cos'),
        (r' pi', ' math.pi'),
        (r', ', ',\n    '),
        (r'\n\[', '\nresults = ['),
        (r'$', '\nreturn np.column_stack((np.column_stack(results[i:(i + 2)]).ravel() for i in range(0, len(results), 2)))'),
        (r'^', 'def jacobian' + str((xyA0, xyA1) + params) + ':\n'),
        (r'\n', '\n    ')]
    text = '\n'.join(lines)
    for old, new in replacements:
        text = re.sub(old, new, text)
    if path:
        with open(path, mode='w') as fp:
            fp.write(text)
    else:
        return text

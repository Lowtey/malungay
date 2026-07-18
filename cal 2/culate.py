import sympy as smp


def solve_indefinite(expression, variable="x"):
    

   
    expression = expression.replace("^", "**")

    
    x = smp.symbols(variable, real=True)

    
    f = smp.sympify(expression)

    
    answer = smp.integrate(f, x)

    return answer


def solve_definite(expression, variable="x", lower=0, upper=1):
    """
    Solves a definite integral.
    """

    expression = expression.replace("^", "**")

    x = smp.symbols(variable, real=True)

    f = smp.sympify(expression)

    answer = smp.integrate(f, (x, lower, upper))

    return answer
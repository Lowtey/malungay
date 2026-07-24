import sympy as smp
from sympy.parsing.sympy_parser import (
    parse_expr,
    standard_transformations,
    implicit_multiplication_application,
)

TRANSFORMATIONS = standard_transformations + (implicit_multiplication_application,)


def _convert_log_notation(expr_str):
    result = []
    i = 0
    n = len(expr_str)
    while i < n:
        if expr_str[i:i + 3] == "log":
            prev_char = expr_str[i - 1] if i > 0 else ""
            next_is_paren = i + 3 < n and expr_str[i + 3] == "("
            if not prev_char.isalnum() and next_is_paren:
                depth = 0
                j = i + 3
                start_inner = j + 1
                while j < n:
                    if expr_str[j] == "(":
                        depth += 1
                    elif expr_str[j] == ")":
                        depth -= 1
                        if depth == 0:
                            break
                    j += 1
                inner = expr_str[start_inner:j]
                result.append(f"(log({inner})/log(10))")
                i = j + 1
                continue
        result.append(expr_str[i])
        i += 1
    return "".join(result)


class IntegralSolver:

    @staticmethod
    def parse_expression(expression):
        expression = str(expression).replace("^", "**")
        expression = _convert_log_notation(expression)
        return parse_expr(
            expression,
            local_dict={"pi": smp.pi, "ln": smp.log, "e": smp.E},
            transformations=TRANSFORMATIONS,
        )

    @staticmethod
    def parse_bound(value_str):
        value_str = str(value_str).strip().replace("^", "**")
        value_str = _convert_log_notation(value_str)
        return parse_expr(
            value_str,
            local_dict={"pi": smp.pi, "ln": smp.log, "e": smp.E},
            transformations=TRANSFORMATIONS,
        )

    @staticmethod
    def _match_direct_form(expr, x):
        num, den = smp.fraction(smp.together(expr))
        if x in num.free_symbols:
            return None
        if not den.is_polynomial(x):
            return None
        poly = smp.Poly(den, x)
        if poly.degree() != 2:
            return None
        A, B, C = poly.all_coeffs()
        if A == 1 and B == 0 and C > 0:
            a = smp.sqrt(C)
            return (num / a) * smp.atan(x / a), f"Direct Inverse Tangent Form: ∫ k/(x² + a²) dx where a = {a}"
        return None

    @staticmethod
    def _match_substitution_form(expr, x):
        expr_can = smp.cancel(expr)
        num, den = smp.fraction(expr_can)
        den_expanded = smp.expand(den)

        if not den_expanded.is_Add:
            return None

        terms = list(den_expanded.args)
        if smp.Integer(1) not in terms:
            return None

        terms_copy = list(terms)
        terms_copy.remove(smp.Integer(1))

        if len(terms_copy) != 1:
            return None

        square = terms_copy[0]
        if not isinstance(square, smp.Pow) or square.exp != 2:
            return None

        g = square.base
        dg = smp.simplify(smp.diff(g, x))

        if smp.simplify(num - dg) == 0:
            return smp.atan(g), f"u-Substitution Pattern: ∫ f'(x)/(1 + [f(x)]²) dx where u = {g}"
        return None

    @staticmethod
    def solve_inverse_tangent(expression, variable="x"):
        x = smp.Symbol(variable)
        expr = IntegralSolver.parse_expression(expression)

        # 1. Try matching direct form (e.g., 1 / (x^2 + 9))
        direct_result = IntegralSolver._match_direct_form(expr, x)
        if direct_result:
            return direct_result[0]

        # 2. Try matching u-substitution form (e.g., f'(x) / (1 + f(x)^2))
        sub_result = IntegralSolver._match_substitution_form(expr, x)
        if sub_result:
            return sub_result[0]

        raise ValueError(
            "Error: Not a recognized inverse tangent form.\n"
            "Supports direct form (x² + a²) and simple substitution f'(x)/(1 + f(x)²).\n"
            "Expressions requiring completing the square are rejected."
        )

    @staticmethod
    def detect_formula(expression, variable="x"):
        try:
            expr = IntegralSolver.parse_expression(expression)
            x = smp.Symbol(variable)

            res_direct = IntegralSolver._match_direct_form(expr, x)
            if res_direct:
                return res_direct[1]

            res_sub = IntegralSolver._match_substitution_form(expr, x)
            if res_sub:
                return res_sub[1]
        except Exception:
            return None

        return None
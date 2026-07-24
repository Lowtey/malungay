"""
Non-Elementary Integral Solver Module
Complements solve.py by handling functions with non-elementary antiderivatives
via Special Functions, Taylor Series Approximations, and Adaptive Numerical Quadrature.
"""

import numpy as np
from scipy import integrate
import sympy as smp
from solve import IntegralSolver


class NonElementarySolver:

    def __init__(self, expr_str, var_str="x"):
        self.var_str = var_str
        self.var_symbol = smp.Symbol(var_str)
        self.raw_expr_str = expr_str
        self.f_expr = IntegralSolver.parse_expression(expr_str)

    def solve_symbolic(self, series_order=6):
        """Attempts symbolic integration.

        If non-elementary, checks for special functions or falls back to
        Taylor Series expansion.
        """
        x = self.var_symbol
        antiderivative = smp.integrate(self.f_expr, x)

        # 1. Check if SymPy left an unevaluated Integral object (True Non-Elementary)
        if antiderivative.has(smp.Integral):
            series_expansion = smp.series(
                self.f_expr, x, 0, n=series_order
            ).removeO()
            approx_antiderivative = smp.integrate(series_expansion, x)

            return {
                "status": "Non-Elementary (Series Approximated)",
                "exact": None,
                "special_form": None,
                "series_approx": approx_antiderivative,
                "order": series_order,
            }

        # 2. Check if the result uses Special Functions (erf, Si, Ci, Ei, li, Fresnel, etc.)
        special_functions = (
            smp.erf,
            smp.erfi,
            smp.Si,
            smp.Ci,
            smp.Ei,
            smp.li,
            smp.fresnelc,
            smp.fresnels,
        )
        has_special = any(
            antiderivative.has(func) for func in special_functions
        )

        return {
            "status": (
                "Special Non-Elementary Closed-Form"
                if has_special
                else "Elementary"
            ),
            "exact": antiderivative,
            "special_form": antiderivative if has_special else None,
            "series_approx": None,
        }

    def evaluate_definite_numerical(self, lower_bound, upper_bound):
        """Uses scipy.integrate.quad (Gauss-Kronrod adaptive quadrature) for

        accurate numerical evaluation of non-elementary definite integrals.
        """
        x = self.var_symbol
        f_num = smp.lambdify(x, self.f_expr, modules=["numpy", "scipy"])

        result, abs_error = integrate.quad(f_num, lower_bound, upper_bound)

        return {
            "value": result,
            "estimated_error": abs_error,
            "bounds": (lower_bound, upper_bound),
        }


# ==========================================
# SELF-TEST (Runs if you execute non_elementary.py directly)
# ==========================================
if __name__ == "__main__":
    test_cases = [
        "exp(-x^2)",  # Error Function: erf(x)
        "sin(x) / x",  # Sine Integral: Si(x)
        "1 / log(x)",  # Logarithmic Integral: li(x)
        "exp(x) / x",  # Exponential Integral: Ei(x)
        "sin(x^2)",  # Fresnel Integral: S(x)
    ]

    print("--- TESTING NON-ELEMENTARY SOLVER ---")
    for expr in test_cases:
        solver = NonElementarySolver(expr)
        sym_res = solver.solve_symbolic(series_order=8)
        num_res = solver.evaluate_definite_numerical(1.0, 2.0)

        print(f"\nIntegrand: f(x) = {expr}")
        print(f"Status   : {sym_res['status']}")

        if sym_res["exact"] is not None:
            print(f"Closed   : F(x) = {sym_res['exact']} + C")
        else:
            print(f"Series   : F(x) ≈ {sym_res['series_approx']} + C")

        print(
            f"∫[1, 2]  : ≈ {num_res['value']:.8f} (Error: ±{num_res['estimated_error']:.2e})"
        )
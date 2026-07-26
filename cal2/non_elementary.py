
from solve import IntegralSolver

class NonElementarySolver:
    def __init__(self, expr_str, var_str="x"):
        self.var_str = var_str
        self.raw_expr_str = expr_str
        # Will throw an error and disregard if not arctan or non-elementary
        self.antiderivative = IntegralSolver.solve_inverse_tangent(expr_str, var_str)

    def solve_symbolic(self, series_order=6):
        return {
            "status": "Valid Inverse Tangent Form",
            "exact": self.antiderivative,
            "special_form": None,
            "series_approx": None,
        }
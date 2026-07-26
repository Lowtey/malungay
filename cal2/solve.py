"""
Cal2lator - Strict Inverse Tangent Integral Solver (backend)

By:
- Cyrus Gabriel Ebarat
- Sebastian Nabalan

To be submitted to:
- Engr. Darwin Jone H. Jupiter

DESIGN NOTE
-----------
This solver deliberately does NOT call sympy.integrate(expr, x) as a
first-pass "does it work" check. Plain sympy.integrate() is a general
algorithm that can reach an arctan answer through routes we don't want to
accept here -- e.g. completing the square, or full partial-fraction
decomposition. Instead, this file only recognizes two direct patterns:

  PATTERN A: C / (A*x^2 + B)
      A literal a^2+x^2 shape (up to a constant multiple). No linear term
      is allowed -- if one is present, solving would require completing
      the square, which is out of scope by design.

  PATTERN B: C * g'(x) / (A*g(x)^2 + B)
      A literal derivative-over-(1+f^2) substitution, where g(x) is either
      (a) a named function already present in the expression (sqrt, sin,
          tan, exp, log, ...), or
      (b) an "implied" power substitution, e.g. x^8 in a denominator
          naturally suggests trying g(x) = x^4, even though x^4 never
          literally appears in the original expression.

Anything that only resolves to arctan via completing the square or
partial fractions (e.g. 1/(x^2+4x+13), 1/((x+2)^2+9), 1/(x^2-9)) is
rejected, even though a general-purpose CAS could solve some of those.
"""

import sympy as smp
from sympy.parsing.sympy_parser import (
    parse_expr,
    standard_transformations,
    implicit_multiplication_application,
)

TRANSFORMATIONS = standard_transformations + (implicit_multiplication_application,)


def custom_log(*args):
    """Forces 'log(x)' to default to base 10, while still allowing
    explicit custom bases like 'log(x, 2)'."""
    if len(args) == 1:
        return smp.log(args[0], 10)
    return smp.log(*args)


# 'ln' -> natural log (base e); 'log' -> base 10; 'e' -> Euler's number
CUSTOM_DICT = {
    "pi": smp.pi,
    "e": smp.E,
    "ln": smp.log,
    "log": custom_log,
}


class IntegralSolver:

    # ------------------------------------------------------------------
    # Parsing
    # ------------------------------------------------------------------

    @staticmethod
    def parse_expression(expression):
        """Converts a user-entered string into a SymPy expression.
        Supports implicit multiplication ('3x^3'), 'ln'/'log' distinction,
        and 'e' as Euler's number."""
        expression = str(expression).replace("^", "**")
        return parse_expr(
            expression,
            local_dict=CUSTOM_DICT,
            transformations=TRANSFORMATIONS,
        )

    @staticmethod
    def parse_bound(value_str):
        """Parses a bound string using the same rules as parse_expression."""
        value_str = str(value_str).strip().replace("^", "**")
        return parse_expr(
            value_str,
            local_dict=CUSTOM_DICT,
            transformations=TRANSFORMATIONS,
        )

    # ------------------------------------------------------------------
    # Pattern A: direct C / (A*x^2 + B)
    # ------------------------------------------------------------------

    @staticmethod
    def _match_a2_plus_var2(numerator, denominator, var):
        """Shared matcher: checks numerator/denominator against
        C / (A*var^2 + B), degree-2, NO linear term. Returns the
        antiderivative (in terms of `var`) or None."""
        if not numerator.is_number:
            return None
        if not denominator.is_polynomial(var):
            return None

        poly = smp.Poly(denominator, var)
        if poly.degree() != 2:
            return None

        coeffs = poly.all_coeffs()
        if len(coeffs) != 3:
            return None

        A, B_linear, C = coeffs
        if B_linear != 0:
            return None  # linear term present -> needs completing the square -> reject

        is_A_pos = A.is_positive or (A.is_number and A > 0)
        is_C_pos = C.is_positive or (C.is_number and C > 0)
        if not (is_A_pos and is_C_pos):
            return None

        a_val = smp.sqrt(C / A)
        return smp.simplify((numerator / A) * smp.atan(var / a_val) / a_val)

    # ------------------------------------------------------------------
    # Pattern B: derivative-over-(a^2 + g(x)^2) substitution
    # ------------------------------------------------------------------

    @staticmethod
    def _substitution_candidates(expr, x):
        """Collects literal, named inner sub-expressions: Function calls
        (sin, cos, tan, exp, log, sqrt...) and rational powers of x.
        Deliberately excludes plain shifts like (x+2), since recognizing
        those needs completing-the-square-style algebra, not a direct sub."""
        candidates = set()
        for f in expr.atoms(smp.Function):
            if x in f.free_symbols:
                candidates.add(f)
        for p in expr.atoms(smp.Pow):
            base, exp = p.as_base_exp()
            if base == x and exp != 1 and exp.is_number:
                candidates.add(p)
        return candidates

    @staticmethod
    def _implied_power_substitution(expr, x):
        """Detects denominators of the exact form A*x^(2k) + B (only those
        two terms) and proposes g = x^k, even if x^k never literally
        appears (e.g. 3x^3/(x^8+4) implies g = x^4)."""
        numerator, denominator = smp.fraction(expr)
        if not denominator.is_polynomial(x):
            return None
        poly = smp.Poly(denominator, x)
        deg = poly.degree()
        if deg < 4 or deg % 2 != 0:
            return None
        k = deg // 2
        monoms = poly.monoms()
        if len(monoms) != 2:
            return None
        degs_present = sorted(m[0] for m in monoms)
        if degs_present != [0, 2 * k]:
            return None
        return x**k

    @staticmethod
    def _direct_derivative_over_1_plus_f2(expr, x):
        """Attempts expr = C*g'(x) / (A*g(x)^2 + B) for some literal,
        named g(x). Purely structural -- never falls back to a general
        integration algorithm."""
        candidates = IntegralSolver._substitution_candidates(expr, x)
        implied = IntegralSolver._implied_power_substitution(expr, x)
        if implied is not None:
            candidates.add(implied)

        u = smp.Symbol("u")

        for g in candidates:
            dgdx = smp.diff(g, x)
            if dgdx == 0:
                continue
            try:
                # trigsimp is needed here because SymPy often represents a
                # derivative using a DIFFERENT (but equal) trig identity
                # than what appears in the numerator -- e.g. d/dx[tan(x)]
                # is stored as tan(x)**2+1, not sec(x)**2, even though
                # they're identical by the Pythagorean identity.
                #
                # ORDER MATTERS: trigsimp must run on the RAW division,
                # before cancel(). Calling cancel() first expands the
                # denominator into pure tan(x) powers, which destroys the
                # sec/tan structure trigsimp relies on to recognize the
                # identity -- trigsimp on the already-cancelled form fails.
                ratio_raw = expr / dgdx
                ratio = smp.trigsimp(ratio_raw)
                ratio = smp.cancel(ratio)
                ratio_u = smp.cancel(ratio.subs(g, u))
            except Exception:
                continue

            if x in ratio_u.free_symbols:
                continue  # g wasn't literally/cleanly present -> not a direct sub

            num_u, den_u = smp.fraction(ratio_u)
            num_u = smp.expand(num_u)
            den_u = smp.expand(den_u)
            antideriv_u = IntegralSolver._match_a2_plus_var2(num_u, den_u, u)
            if antideriv_u is not None:
                return smp.simplify(antideriv_u.subs(u, g))

        return None

    # ------------------------------------------------------------------
    # Public solving interface
    # ------------------------------------------------------------------

    @staticmethod
    def solve_inverse_tangent(expression, variable="x"):
        """Solves an integral ONLY if it matches a direct arctan pattern
        (Pattern A or Pattern B above). Raises ValueError otherwise --
        including for integrals a general CAS could solve through other
        means (completing the square, partial fractions) and for
        integrals with no elementary closed form at all.
        """
        x = smp.Symbol(variable)
        expr = IntegralSolver.parse_expression(expression)

        numerator, denominator = smp.fraction(expr)

        antiderivative = IntegralSolver._match_a2_plus_var2(numerator, denominator, x)
        if antiderivative is None:
            antiderivative = IntegralSolver._direct_derivative_over_1_plus_f2(expr, x)

        if antiderivative is None:
            raise ValueError(
                "Error: Not a direct inverse tangent (arctan) pattern.\n"
                "This solver only accepts 1/(a²+x²) or a direct\n"
                "derivative-over-(a²+f(x)²) substitution.\nDisregarded."
            )

        return antiderivative

    @staticmethod
    def detect_formula(expression, variable="x"):
        """Convenience check: returns a short label if the expression is
        solvable by this strict solver, else None."""
        try:
            IntegralSolver.solve_inverse_tangent(expression, variable)
            return "Valid Inverse Tangent Form"
        except Exception:
            return None
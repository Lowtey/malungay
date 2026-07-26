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

    # Explicit Pythagorean identity rewrites, applied before cancellation.
    # This is more reliable than smp.trigsimp() alone -- trigsimp tries
    # several internal strategies and can land on a DIFFERENT equivalent
    # form (e.g. rewriting everything in sin/cos) that doesn't actually
    # help the cancellation, depending on call order. Explicit substitution
    # guarantees the numerator and the derivative end up expressed in the
    # SAME underlying function before dividing.
    _PYTHAGOREAN_SUBS = {
        smp.sec: lambda x: {smp.sec(x)**2: smp.tan(x)**2 + 1},
        smp.tan: lambda x: {smp.sec(x)**2: smp.tan(x)**2 + 1},
        smp.csc: lambda x: {smp.csc(x)**2: smp.cot(x)**2 + 1},
        smp.cot: lambda x: {smp.csc(x)**2: smp.cot(x)**2 + 1},
    }

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

            # If g is a trig function with a known Pythagorean-identity
            # partner (sec<->tan, csc<->cot), prepare an identity-rewrite
            # of BOTH the numerator and the derivative as a FALLBACK.
            identity_subs = {}
            g_func = g.func if hasattr(g, "func") else None
            if g_func in IntegralSolver._PYTHAGOREAN_SUBS:
                identity_subs = IntegralSolver._PYTHAGOREAN_SUBS[g_func](x)

            # Attempt 1: plain division, no identity rewriting. This
            # correctly handles cases where the numerator ALREADY equals
            # dgdx exactly (e.g. sec(x)*tan(x)/(1+sec(x)**2), where
            # dgdx = sec(x)*tan(x) matches the numerator literally) --
            # forcing an identity rewrite here would needlessly convert
            # the denominator into different terms and break the match.
            attempts = [(expr, dgdx)]

            # Attempt 2 (fallback): identity-normalized, for cases where
            # the numerator and dgdx are equal only via a Pythagorean
            # identity, not literally (e.g. sec(x)**2 vs tan(x)**2+1).
            if identity_subs:
                attempts.append((expr.subs(identity_subs), dgdx.subs(identity_subs)))

            ratio_u = None
            for expr_try, dgdx_try in attempts:
                try:
                    ratio = smp.cancel(expr_try / dgdx_try)
                    candidate_ratio_u = smp.cancel(ratio.subs(g, u))
                except Exception:
                    continue
                if x not in candidate_ratio_u.free_symbols:
                    ratio_u = candidate_ratio_u
                    break  # found a clean substitution, stop trying

            if ratio_u is None:
                continue  # neither attempt cleared x -> not a direct sub

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
        try:
            expr = IntegralSolver.parse_expression(expression)
        except Exception:
            raise ValueError(
                "Error: Could not parse this expression.\n"
                "Check for unbalanced parentheses, missing operators,\n"
                "or invalid syntax.\nDisregarded."
            )

        # Guard: if the integrand trivially collapses to a constant via a
        # trig/algebraic identity (e.g. sec(x)**2/(tan(x)**2+1) === 1), it
        # can NEVER be a genuine arctan candidate -- its real antiderivative
        # is just (constant)*x. Without this check, the Pythagorean-identity
        # rewrite used in Pattern B can make such a numerator cancel exactly
        # against the denominator and get misread as a valid a^2+u^2 match.
        collapsed = smp.trigsimp(smp.simplify(expr))
        if x not in collapsed.free_symbols:
            raise ValueError(
                "Error: This integrand is a disguised constant (collapses\n"
                "to a fixed value via a trig/algebraic identity), so it\n"
                "cannot be an inverse tangent (arctan) integral.\nDisregarded."
            )

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
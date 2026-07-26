import customtkinter as ctk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from solve import IntegralSolver
import numpy as np
import sympy as smp

ctk.set_appearance_mode("System")
ctk.set_default_color_theme("blue")


class Cal2latorApp(ctk.CTk):

    def __init__(self):
        super().__init__()

        self.title("Cal2lator - Strict Inverse Tangent Solver")
        self.geometry("1100x800")
        self.resizable(True, True)

        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=2)
        self.grid_rowconfigure(0, weight=1)

        self.control_frame = ctk.CTkScrollableFrame(self, corner_radius=10)
        self.control_frame.grid(row=0, column=0, padx=15, pady=15, sticky="nsew")

        self.graph_frame = ctk.CTkFrame(self, corner_radius=10)
        self.graph_frame.grid(row=0, column=1, padx=15, pady=15, sticky="nsew")

        title = ctk.CTkLabel(
            self.control_frame, text="CAL2LATOR", font=("Segoe UI", 28, "bold")
        )
        title.pack(pady=(15, 5))

        self._build_control_panel()
        self._add_credits()
        self.canvas = None

    def _build_control_panel(self):
        ctk.CTkLabel(
            self.control_frame, text="Enter Integrand f(x):", font=("Arial", 14)
        ).pack(anchor="w", padx=15, pady=(10, 0))

        self.input_var = ctk.StringVar()
        self.input_var.trace_add("write", self.update_live_preview)

        self.function_entry = ctk.CTkEntry(
            self.control_frame,
            width=280,
            height=40,
            corner_radius=10,
            textvariable=self.input_var,
            placeholder_text="Example: 1 / (x^2 + 9)"
        )
        self.function_entry.pack(anchor="w", padx=15, pady=(6, 4))

        self.preview_label = ctk.CTkLabel(
            self.control_frame,
            text="Preview: f(x) = ...",
            font=("Arial", 13, "italic"),
            text_color="#3B82F6"
        )
        self.preview_label.pack(anchor="w", padx=18, pady=(0, 10))

        ctk.CTkLabel(
            self.control_frame,
            text="Variable of Integration:",
            font=("Arial", 14),
        ).pack(anchor="w", padx=15, pady=(5, 0))

        self.var_str_var = ctk.StringVar(value="x")
        self.var_str_var.trace_add("write", self.update_live_preview)

        self.variable_entry = ctk.CTkEntry(
            self.control_frame,
            width=280,
            height=40,
            corner_radius=10,
            textvariable=self.var_str_var,
            placeholder_text="x"
        )
        self.variable_entry.pack(anchor="w", padx=15, pady=(5, 10))

        ctk.CTkLabel(
            self.control_frame, text="Integral Type:", font=("Arial", 14)
        ).pack(anchor="w", padx=15, pady=(5, 0))
        self.mode = ctk.StringVar(value="indefinite")

        self.radio1 = ctk.CTkRadioButton(
            self.control_frame,
            text="Indefinite Integral",
            variable=self.mode,
            value="indefinite",
            command=self.toggle_bounds_visibility,
        )
        self.radio2 = ctk.CTkRadioButton(
            self.control_frame,
            text="Definite Integral",
            variable=self.mode,
            value="definite",
            command=self.toggle_bounds_visibility,
        )
        self.radio1.pack(anchor="w", padx=15, pady=5)
        self.radio2.pack(anchor="w", padx=15, pady=5)

        self.bounds_frame = ctk.CTkFrame(
            self.control_frame, fg_color="transparent"
        )

        ctk.CTkLabel(self.bounds_frame, text="Lower Bound:").pack(anchor="w")
        self.lower_entry = ctk.CTkEntry(
            self.bounds_frame, width=280, height=40, corner_radius=10, placeholder_text="0"
        )
        self.lower_entry.pack(anchor="w", pady=(0, 5))

        ctk.CTkLabel(self.bounds_frame, text="Upper Bound:").pack(anchor="w")
        self.upper_entry = ctk.CTkEntry(
            self.bounds_frame, width=280, height=40, corner_radius=10, placeholder_text="1"
        )
        self.upper_entry.pack(anchor="w", pady=(0, 10))

        self.bounds_frame.pack_forget()

        self.solve_button = ctk.CTkButton(
            self.control_frame,
            text="Solve & Graph",
            width=280,
            height=45,
            corner_radius=12,
            font=("Segoe UI", 16, "bold"),
            command=self.solve_and_plot,
        )
        self.solve_button.pack(anchor="w", padx=15, pady=15)

        result_frame = ctk.CTkFrame(self.control_frame, corner_radius=12)
        result_frame.pack(fill="x", padx=15, pady=(5, 10))

        ctk.CTkLabel(
            result_frame, text="Results", font=("Arial", 15, "bold")
        ).pack(anchor="w", padx=10, pady=(10, 5))

        self.result_box = ctk.CTkTextbox(
            result_frame,
            font=("Consolas", 14),
            wrap="word",
            height=250,
            corner_radius=8,
            fg_color="#1e1e1e",
            text_color="#f3f4f6"
        )
        self.result_box.pack(fill="x", padx=10, pady=(0, 10))
        self.update_result_box("Waiting for input...")

    def update_result_box(self, text, is_error=False):
        self.result_box.configure(state="normal")
        self.result_box.delete("0.0", "end")
        self.result_box.insert("0.0", text)
        if is_error:
            self.result_box.configure(text_color="#EF4444")
        else:
            self.result_box.configure(text_color="#f3f4f6")
        self.result_box.configure(state="disabled")

    def append_result_box(self, text):
        self.result_box.configure(state="normal")
        self.result_box.insert("end", text)
        self.result_box.configure(state="disabled")

    def _add_credits(self):
        credits_text = (
            "By:\n"
            "- Cyrus Gabriel Ebarat\n"
            "- Sebastien Nabalan\n\n"
            "To be submitted to:\n"
            "- Engr. Darwin Jone H. Jupiter"
        )
        ctk.CTkLabel(
            self.control_frame,
            text=credits_text,
            font=("Arial", 11, "italic"),
            text_color="#888888",
            justify="left"
        ).pack(anchor="w", padx=15, pady=(15, 10))

    def update_live_preview(self, *args):
        raw_text = self.input_var.get().strip()
        var_text = self.var_str_var.get().strip() or "x"

        if not raw_text:
            self.preview_label.configure(
                text=f"Preview: f({var_text}) = ...", text_color="#3B82F6"
            )
            return

        try:
            IntegralSolver.parse_expression(raw_text)
            formatted = (
                raw_text.replace("**", "^")
                .replace("*", " · ")
                .replace("atan", "tan⁻¹")
                .replace("arctan", "tan⁻¹")
                .replace("sqrt", "√")
                .replace("pi", "π")
            )
            self.preview_label.configure(
                text=f"Preview: f({var_text}) = {formatted}",
                text_color="#3B82F6",
            )
        except Exception:
            self.preview_label.configure(
                text="Preview: Invalid math syntax", text_color="#EF4444"
            )

    def toggle_bounds_visibility(self):
        if self.mode.get() == "definite":
            self.bounds_frame.pack(anchor="w", padx=15, pady=5, after=self.radio2)
        else:
            self.bounds_frame.pack_forget()

    def solve_and_plot(self):
        raw_expr = self.function_entry.get().strip()
        var_str = self.variable_entry.get().strip()

        if not raw_expr or not var_str or not var_str.isalpha():
            self.update_result_box("Error: Enter a valid integrand and variable.", is_error=True)
            return

        raw_expr = raw_expr.replace("^", "**")

        try:
            var_symbol = smp.Symbol(var_str)
            f_expr = IntegralSolver.parse_expression(raw_expr)
            antiderivative = IntegralSolver.solve_inverse_tangent(raw_expr, var_str)

            clean_antideriv = str(antiderivative).replace("pi", "π").replace("**", "^").replace("*", "·")
            clean_raw = raw_expr.replace("**", "^").replace("*", "·")

            message = "--- INTEGRATION ---\n\n"
            message += f"Integrand f({var_str}):\n{clean_raw}\n\n"
            message += f"Antiderivative F({var_str}):\n{clean_antideriv} + C\n\n"

            if self.mode.get() == "indefinite":
                message += "Status:\nValid Arctan Form\n"
                self.update_result_box(message)
                self.plot_graphs(f_expr, antiderivative, var_symbol, is_definite=False)
            
            else:
                lower_str = self.lower_entry.get().strip() or "0"
                upper_str = self.upper_entry.get().strip() or "1"
                
                lower_val = IntegralSolver.parse_bound(lower_str)
                upper_val = IntegralSolver.parse_bound(upper_str)

                # Detect flipped bounds (lower > upper) for display purposes.
                # The math itself (F(upper)-F(lower)) already handles this
                # correctly and automatically -- no swap needed -- this is
                # purely an informational note for the user.
                bounds_flipped = False
                try:
                    bounds_flipped = bool(lower_val > upper_val)
                except TypeError:
                    pass  # symbolic bounds that can't be compared -- skip the note

                # Pure mathematical evaluation. Works perfectly even if a > b.
                F_upper = antiderivative.subs(var_symbol, upper_val)
                F_lower = antiderivative.subs(var_symbol, lower_val)
                definite_ans = smp.simplify(F_upper - F_lower)

                # Safely parse numeric bounds for visual plotting ONLY. 
                # Prevents program crash when symbolic bounds (like pi) are used.
                try:
                    l_num = float(lower_val.evalf())
                    u_num = float(upper_val.evalf())
                    plot_lower = min(l_num, u_num)
                    plot_upper = max(l_num, u_num)
                except Exception:
                    plot_lower, plot_upper = 0, 1

                clean_ans = str(definite_ans).replace("pi", "π").replace("**", "^").replace("*", "·")
                numeric_ans = definite_ans.evalf(6)

                # Numerical sanity check: F(b)-F(a) is only valid if F is
                # CONTINUOUS on [a,b]. Some accepted antiderivatives (e.g.
                # atan(tan(x)/2)/2) have a removable jump discontinuity
                # wherever the inner substitution function (tan, cot, ...)
                # has a vertical asymptote. If the bounds straddle one of
                # those points, direct substitution silently gives a wrong
                # answer. We catch this by comparing against a real
                # numerical integration of f(x) itself over the same
                # interval, using only numpy (already imported) -- no new
                # dependencies.
                discontinuity_warning = ""
                try:
                    f_num = smp.lambdify(var_symbol, f_expr, modules=["numpy"])
                    n_samples = 20000
                    t = np.linspace(l_num, u_num, n_samples)
                    with np.errstate(divide="ignore", invalid="ignore"):
                        y = f_num(t)
                    y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
                    # Manual trapezoidal rule -- np.trapz() was removed in
                    # NumPy 2.x (renamed to np.trapezoid), so this avoids
                    # depending on which NumPy version is installed.
                    numeric_check = float(np.sum((y[:-1] + y[1:]) * np.diff(t)) / 2.0)
                    symbolic_val = float(numeric_ans)
                    if abs(numeric_check - symbolic_val) > max(1e-3, abs(numeric_check) * 0.02):
                        discontinuity_warning = (
                            "\n\n⚠ WARNING: The antiderivative F(x) appears to be\n"
                            "discontinuous somewhere inside [{}, {}] (likely where\n"
                            "the substitution function has a vertical asymptote).\n"
                            "Direct F(b)-F(a) evaluation may be WRONG here.\n"
                            "Numerical cross-check estimate:\n≈ {:.6f}"
                        ).format(lower_str, upper_str, numeric_check)
                except Exception:
                    pass  # sanity check itself failed -- don't block the normal result

                message += "--- DEFINITE EVALUATION ---\n\n"
                message += f"Limits: [{lower_str}, {upper_str}]\n\n"
                if bounds_flipped:
                    message += "Note: Bounds were flipped (lower > upper).\n\n"
                message += f"Exact Answer:\n{clean_ans}\n\n"
                message += f"Numerical Value:\n≈ {numeric_ans}"
                message += discontinuity_warning

                self.update_result_box(message)
                self.plot_graphs(
                    f_expr, antiderivative, var_symbol, is_definite=True,
                    bounds=(plot_lower, plot_upper)
                )

        except Exception as e:
            self.update_result_box(str(e), is_error=True)
            if self.canvas:
                self.canvas.get_tk_widget().destroy()
                self.canvas = None

    def plot_graphs(self, f_expr, F_expr, var_symbol, is_definite=False, bounds=None):
        try:
            f_num = smp.lambdify(var_symbol, f_expr, modules=["numpy"])
            F_num = smp.lambdify(var_symbol, F_expr, modules=["numpy"])

            if is_definite and bounds:
                a, b = bounds
                margin = max(abs(b - a) * 0.5, 1.0)
                t = np.linspace(a - margin, b + margin, 400)
            else:
                t = np.linspace(-10, 10, 400)

            with np.errstate(divide="ignore", invalid="ignore"):
                y_f = f_num(t)
                y_F = F_num(t)

            if not isinstance(y_f, np.ndarray):
                y_f = np.full_like(t, y_f)
            if not isinstance(y_F, np.ndarray):
                y_F = np.full_like(t, y_F)

            if self.canvas:
                self.canvas.get_tk_widget().destroy()

            fig, ax = plt.subplots(figsize=(5, 5), dpi=100)
            fig.patch.set_facecolor("#2b2b2b")
            ax.set_facecolor("#1e1e1e")

            ax.plot(t, y_f, label=f"f({var_symbol})", color="#1f77b4", lw=2)
            ax.plot(t, y_F, label=f"F({var_symbol})", color="#ff7f0e", linestyle="--", lw=2)

            if is_definite and bounds:
                a, b = bounds
                t_shade = np.linspace(a, b, 200)
                with np.errstate(divide="ignore", invalid="ignore"):
                    y_shade = f_num(t_shade)
                if not isinstance(y_shade, np.ndarray):
                    y_shade = np.full_like(t_shade, y_shade)
                ax.fill_between(t_shade, y_shade, alpha=0.3, color="#1f77b4", label="Area")

            ax.axhline(0, color="gray", linewidth=0.8, linestyle=":")
            ax.axvline(0, color="gray", linewidth=0.8, linestyle=":")
            ax.tick_params(colors="white")
            ax.grid(True, color="#333333", linestyle="--")
            ax.legend(facecolor="#2b2b2b", edgecolor="none", labelcolor="white")
            fig.tight_layout()

            self.canvas = FigureCanvasTkAgg(fig, master=self.graph_frame)
            self.canvas.draw()
            self.canvas.get_tk_widget().pack(fill="both", expand=True, padx=5, pady=5)
            
        except Exception as e:
            if self.canvas:
                self.canvas.get_tk_widget().destroy()
                self.canvas = None
            self.append_result_box(f"\n\n[Plotting Error: {str(e)}]")


if __name__ == "__main__":
    app = Cal2latorApp()
    app.mainloop()
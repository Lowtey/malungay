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

        self.title("Cal2lator - Inverse Tangent Solver")
        self.geometry("1100x800")
        self.resizable(True, True)

        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=2)
        self.grid_rowconfigure(0, weight=1)

        self.control_frame = ctk.CTkScrollableFrame(self, corner_radius=10)
        self.control_frame.grid(
            row=0, column=0, padx=15, pady=15, sticky="nsew"
        )

        self.graph_frame = ctk.CTkFrame(self, corner_radius=10)
        self.graph_frame.grid(row=0, column=1, padx=15, pady=15, sticky="nsew")

        title = ctk.CTkLabel(
            self.control_frame,
            text="CAL2LATOR",
            font=("Segoe UI", 28, "bold")
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
        self.upper_entry.pack(anchor="w", pady=(0, 5))

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

        result_frame = ctk.CTkFrame(
            self.control_frame,
            corner_radius=12
        )
        result_frame.pack(
            fill="x",
            padx=15,
            pady=(5, 10)
        )

        ctk.CTkLabel(
            result_frame,
            text="Result",
            font=("Arial", 15, "bold")
        ).pack(anchor="w", padx=10, pady=(10, 5))

        self.result_label = ctk.CTkLabel(
            result_frame,
            text="Waiting for input...",
            font=("Consolas", 13),
            text_color=("#1f2937", "#f3f4f6"),
            wraplength=250,
            justify="left"
        )
        self.result_label.pack(
            anchor="w",
            padx=10,
            pady=(0, 10)
        )

    def _add_credits(self):
        credits_text = (
            "By:\n"
            "- Cyrus Gabriel Ebarat\n"
            "- Sebastian Nabalan\n\n"
            "To be submitted to:\n"
            "- Engr. Darwin Jone H. Jupiter"
        )
        credits_label = ctk.CTkLabel(
            self.control_frame,
            text=credits_text,
            font=("Arial", 11, "italic"),
            text_color="#888888",
            justify="left"
        )
        credits_label.pack(anchor="w", padx=15, pady=(15, 10))

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
                text="Preview: Invalid math syntax",
                text_color="#EF4444",
            )

    def toggle_bounds_visibility(self):
        if self.mode.get() == "definite":
            self.bounds_frame.pack(anchor="w", padx=15, pady=5, after=self.radio2)
        else:
            self.bounds_frame.pack_forget()

    def solve_and_plot(self):
        raw_expr = self.function_entry.get().strip()
        var_str = self.variable_entry.get().strip()

        if not raw_expr:
            self.result_label.configure(
                text="Error: Integrand cannot be empty.", text_color="#EF4444"
            )
            return

        if not var_str or not var_str.isalpha():
            self.result_label.configure(
                text="Error: Variable must be a valid letter (e.g., x, t).",
                text_color="#EF4444",
            )
            return

        raw_expr = raw_expr.replace("^", "**")

        try:
            var_symbol = smp.Symbol(var_str)
            f_expr = IntegralSolver.parse_expression(raw_expr)

            try:
                antiderivative = IntegralSolver.solve_inverse_tangent(
                    raw_expr,
                    var_str
                )
            except ValueError as e:
                self.result_label.configure(
                    text=str(e),
                    text_color="#EF4444"
                )
                if self.canvas:
                    self.canvas.get_tk_widget().destroy()
                    self.canvas = None
                return

            if self.mode.get() == "indefinite":
                detected = IntegralSolver.detect_formula(raw_expr, variable=var_str)
                clean_antideriv = str(antiderivative).replace("pi", "π")

                message = (
                    "━━━━━━━━━━━━━━━━━━━━━━\n"
                    "        RESULT        \n"
                    "━━━━━━━━━━━━━━━━━━━━━━\n\n"
                )
                message += f"Input:\n∫ {raw_expr} d{var_str}\n\n"

                if detected:
                    message += f"Pattern Match:\n{detected}\n\n"

                message += f"Solution:\n{clean_antideriv} + C\n\n"
                message += "Status:\n Valid Inverse Tangent Form"

                self.result_label.configure(
                    text=message,
                    text_color=("#1f2937", "#f3f4f6")
                )

                self.plot_graphs(
                    f_expr,
                    antiderivative,
                    var_symbol,
                    is_definite=False
                )

            else:
                lower_str = self.lower_entry.get().strip() or "0"
                upper_str = self.upper_entry.get().strip() or "1"

                try:
                    lower_val = IntegralSolver.parse_bound(lower_str)
                    upper_val = IntegralSolver.parse_bound(upper_str)
                except Exception:
                    self.result_label.configure(
                        text="Error: Could not parse lower or upper bound syntax.",
                        text_color="#EF4444",
                    )
                    if self.canvas:
                        self.canvas.get_tk_widget().destroy()
                        self.canvas = None
                    return

                F_upper = antiderivative.subs(var_symbol, upper_val)
                F_lower = antiderivative.subs(var_symbol, lower_val)
                definite_ans = smp.simplify(F_upper - F_lower)

                plot_lower, plot_upper = lower_val, upper_val
                flipped = False
                if lower_val > upper_val:
                    plot_lower, plot_upper = upper_val, lower_val
                    flipped = True

                clean_ans = str(definite_ans).replace("pi", "π")
                numeric_ans = definite_ans.evalf(6)

                orig_lower_str = lower_str.replace("pi", "π")
                orig_upper_str = upper_str.replace("pi", "π")

                message = (
                    "━━━━━━━━━━━━━━━━━━━━━━\n"
                    "    DEFINITE RESULT   \n"
                    "━━━━━━━━━━━━━━━━━━━━━━\n\n"
                )
                message += f"Input:\n∫[{orig_lower_str}, {orig_upper_str}] {raw_expr} d{var_str}\n\n"

                if flipped:
                    message += "Note: Bounds were flipped (lower > upper).\nResult sign adjusted.\n\n"

                message += f"Exact Answer:\n{clean_ans}\n\n"
                message += f"Numerical Value:\n≈ {numeric_ans}"

                self.result_label.configure(
                    text=message,
                    text_color=("#1f2937", "#f3f4f6")
                )

                self.plot_graphs(
                    f_expr,
                    antiderivative,
                    var_symbol,
                    is_definite=True,
                    bounds=(float(plot_lower.evalf()), float(plot_upper.evalf())),
                )

        except Exception as e:
            self.result_label.configure(
                text=f"Error: {str(e)}",
                text_color="#EF4444"
            )
            if self.canvas:
                self.canvas.get_tk_widget().destroy()
                self.canvas = None

    def plot_graphs(
        self, f_expr, F_expr, var_symbol, is_definite=False, bounds=None
    ):
        extra_symbols = (f_expr.free_symbols | F_expr.free_symbols) - {var_symbol}
        if extra_symbols:
            if self.canvas:
                self.canvas.get_tk_widget().destroy()
                self.canvas = None
            current_text = self.result_label.cget("text")
            self.result_label.configure(
                text=f"{current_text}\n\n[Plotting Skipped: Parameter(s): {', '.join(str(s) for s in extra_symbols)}]"
            )
            return

        try:
            f_num = smp.lambdify(var_symbol, f_expr, modules=["numpy"])
            F_num = smp.lambdify(var_symbol, F_expr, modules=["numpy"])

            if is_definite and bounds:
                a, b = bounds

                plot_limit = 10
                if np.isneginf(a) or np.isposinf(a):
                    a = -plot_limit
                if np.isposinf(b) or np.isneginf(b):
                    b = plot_limit

                margin = max(abs(b - a) * 0.5, 1.0)
                t = np.linspace(a - margin, b + margin, 400)
            else:
                t = np.linspace(-2 * np.pi, 2 * np.pi, 400)

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

            ax.plot(t, y_f, label=f"Integrand: f({var_symbol})", color="#1f77b4", lw=2)
            ax.plot(
                t,
                y_F,
                label=f"Solution: F({var_symbol})",
                color="#ff7f0e",
                linestyle="--",
                lw=2,
            )

            if is_definite and bounds:
                a, b = bounds

                if np.isneginf(a) or np.isposinf(a):
                    a = -10
                if np.isposinf(b) or np.isneginf(b):
                    b = 10

                t_shade = np.linspace(a, b, 200)
                with np.errstate(divide="ignore", invalid="ignore"):
                    y_shade = f_num(t_shade)
                if not isinstance(y_shade, np.ndarray):
                    y_shade = np.full_like(t_shade, y_shade)
                ax.fill_between(
                    t_shade,
                    y_shade,
                    alpha=0.3,
                    color="#1f77b4",
                    label="Definite Area",
                )

            ax.axhline(0, color="gray", linewidth=0.8, linestyle=":")
            ax.axvline(0, color="gray", linewidth=0.8, linestyle=":")
            ax.tick_params(colors="white")
            ax.xaxis.label.set_color("white")
            ax.yaxis.label.set_color("white")
            ax.title.set_color("white")
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
            current_text = self.result_label.cget("text")
            self.result_label.configure(
                text=f"{current_text}\n\n[Plotting Error: {str(e)}]"
            )


if __name__ == "__main__":
    app = Cal2latorApp()
    app.mainloop()
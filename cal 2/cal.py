from sympy import symbols, integrate
import matplotlib.pyplot as plt
import numpy as np
import sympy          
import customtkinter


app = customtkinter.CTk()

app.title("Cal2lator")

app.geometry("700x600")
app.resizable(False, False)

function_label = customtkinter.CTkLabel(
    app,
    text="Enter Function:"
)

function_label.pack()
function_entry = customtkinter.CTkEntry(
    app,
    width=350,
    placeholder_text="Example: x**2 + 3*x"
)

function_entry.pack(pady=10)

variable_label = customtkinter.CTkLabel(
    app,
    text="Variable:"
)

variable_label.pack()

variable_entry = customtkinter.CTkEntry(
    app,
    width=80,
    placeholder_text="x"
)

variable_entry.pack(pady=10)

mode = customtkinter.StringVar(value="indefinite")

radio1 = customtkinter.CTkRadioButton(
    app,
    text="Indefinite Integral",
    variable=mode,
    value="indefinite"
)

radio2 = customtkinter.CTkRadioButton(
    app,
    text="Definite Integral",
    variable=mode,
    value="definite"
)

radio1.pack()
radio2.pack()

lower_label = customtkinter.CTkLabel(app,text="Lower Bound")
lower_label.pack()

lower_entry = customtkinter.CTkEntry(app,width=100)
lower_entry.pack()

upper_label = customtkinter.CTkLabel(app,text="Upper Bound")
upper_label.pack()

upper_entry = customtkinter.CTkEntry(app,width=100)
upper_entry.pack()

result_label = customtkinter.CTkLabel(
    app,
    text="Result will appear here.",
    font=("Arial",16)
)

result_label.pack(pady=20)

solve_button = customtkinter.CTkButton(
    app,
    text="Solve"
)

solve_button.pack(pady=20)


app.mainloop()
# x = symbols('x')

# expr = x**2 + 3*x

# answer = integrate(expr, x)

# print("Integral:", answer)

# t = np.linspace(-5, 5, 100)
# y = t**2 + 3*t

# plt.plot(t, y)
# plt.title("y = x² + 3x")
# plt.grid(True)
# plt.show()
import streamlit as st
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from scipy.integrate import quad
import re
import io
import os
import warnings

st.set_page_config(page_title="Studio Integrale Intuitivo", layout="wide")
warnings.filterwarnings('ignore')

# === HELPERS intelligenti per parsing e validazione ===
DEFAULT_FUNCS = [
    ("log(x)/(x*(x-1))", "log(x)/(x*(x-1))"),
    ("eˣ⋅sin(x)", "exp(x)*sin(x)"),
    ("1/sqrt(1-x**2)", "1/sqrt(1-x**2)"),
    ("ln(x+1)/x", "log(x+1)/x"),
    ("x²", "x**2"),
    ("sin(x)/x", "sin(x)/x"),
]
MATH_REPL = [
    (r"(\bln\b)", "log"),  # ln → log
    (r"\^", "**"),
    (r"π", "pi"),
    (r"∞", "oo"),
    (r"\be\b", "E"),
    (r"exp\(([^\)]+)\)", r"exp(\1)"),
    (r"√\s*\(([^)]+)\)", r"sqrt(\1)"),
]

def clean_func_string(s, variable="x"):
    """Sistema la funzione inserita dall'utente in modo robusto."""
    s = s.replace(" ", "")
    for pat, repl in MATH_REPL:
        s = re.sub(pat, repl, s, flags=re.IGNORECASE)
    # Accorcia variabili capitali (X → x)
    s = re.sub(rf"{variable.upper()}", variable, s)
    # Sostituisci pow(x,n) con x**n
    s = re.sub(r"pow\(([^,]+),([^\)]+)\)", r"(\1)**(\2)", s)
    return s

def valid_variable(v):
    v = v.strip()
    if len(v) != 1 or not (v.isalpha()):
        raise ValueError("La variabile dev'essere una singola lettera.")
    return v

def to_sympy_expr(func_str, variable="x"):
    safe_dict = {
        'log': sp.log, 'ln': sp.log, 'exp': sp.exp, 'sqrt': sp.sqrt,
        'sin': sp.sin, 'cos': sp.cos, 'tan': sp.tan,
        'E': sp.E, 'pi': sp.pi, 'oo': sp.oo,
        'abs': sp.Abs
    }
    t = sp.symbols(variable)
    try:
        expr = sp.sympify(func_str, locals={variable: t, **safe_dict})
        return expr, ""
    except Exception as e:
        return None, f"Errore di sintassi: {e}"

def latexify(expr):
    try:
        return sp.latex(expr)
    except Exception:
        return str(expr)

def download_plot(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    buf.seek(0)
    return buf

# === UI ===

st.title("📐 Studio Integrale - Inserisci formula intuitiva!")
st.markdown("""
Immetti la funzione integranda in modo naturale.  
Esempi accettati: `sin(x)/x`, `log(x)/(x*(x-1))`, `exp(-x**2)`, `sqrt(1-x**2)`, `x^2 + ln(x+1)/x`, ...  
""")

st.sidebar.header("🧭 Quick Insert esempi")
for friendly, code in DEFAULT_FUNCS:
    if st.sidebar.button(friendly):
        st.session_state['func_input'] = code

st.sidebar.markdown("""
**Legenda sintattica**  
- log(x), ln(x) = logaritmo naturale  
- exp(x) = esponenziale  
- sqrt(x) = radice quadrata  
- sin(x), cos(x), tan(x)  
- pow(x,2) o x^2 o x**2 (equivalenti)  
- π, ∞ possono essere scritti come pi, oo
- `abs(x)` per valore assoluto
""")

# === PARAM SETUP ===
if 'func_input' not in st.session_state: st.session_state['func_input'] = DEFAULT_FUNCS[0][1]
if 'var_input' not in st.session_state: st.session_state['var_input'] = "x"
if 'x0_input' not in st.session_state: st.session_state['x0_input'] = 1.0
if 'x1_input' not in st.session_state: st.session_state['x1_input'] = 2.0

colA, colB = st.columns([2,1])

with colA:
    func_input = st.text_input("Funzione integranda f(x):", value=st.session_state['func_input'], key="fi")
    variable = st.text_input("Variabile d'integrazione [es: x]:", value=st.session_state['var_input'], key="vi")

with colB:
    st.markdown("**Limiti integrale definito**")
    x0 = st.number_input("Estremo inferiore (x0):", value=st.session_state['x0_input'], key="x0")
    x1 = st.number_input("Estremo superiore (x1):", value=st.session_state['x1_input'], key="x1")
    st.markdown("**Range asse x (visualizzazione grafico):**")
    xmin = st.number_input("Minimo x:", value=-1.0)
    xmax = st.number_input("Massimo x:", value=5.0)

# --- Validazione + Parsing ---
error_msg = ""
try:
    variable = valid_variable(variable)
except ValueError as e:
    error_msg = str(e)

func_cleaned = clean_func_string(func_input, variable=variable)
f_expr, parse_err = to_sympy_expr(func_cleaned, variable=variable)
if parse_err:
    error_msg = parse_err

if not error_msg:
    t = sp.symbols(variable)
    try:
        # Costruzione funzioni
        f_lambd = sp.lambdify(t, f_expr, modules=['numpy'])
        f_deriv = sp.simplify(sp.diff(f_expr, t))
        f_primitive = sp.simplify(sp.integrate(f_expr, t))
        primitive_latex = latexify(f_primitive)
        deriv_latex = latexify(f_deriv)
        expr_latex = latexify(f_expr)
    except Exception as e:
        error_msg = f"Errore: {e}"

# --- UI OUTPUT ---
if error_msg:
    st.error(error_msg)
    with st.expander("Guida sintassi e troubleshooting"):
        st.markdown("""
- Usare una sola variabile, es: `x`
- Logaritmo: `log(x)` o `ln(x)`
- Esponenzial: `exp(x)`
- Potenza: `x^2` oppure `x**2` oppure `pow(x,2)`
- Attenzione a parentesi, / a divisioni (`(x+1)/(x^2)`), ...  
Se l'integrale è molto difficile può fallire il calcolo simbolico.
        """)
else:
    col1, col2, col3 = st.columns(3)
    with col1:
        st.latex(r"f("+variable+") = " + expr_latex)
    with col2:
        st.latex(r"f'("+variable+") = " + deriv_latex)
    with col3:
        st.latex(r"\int " + expr_latex + f" d{variable} = " + primitive_latex + r" + C")

    st.markdown("#### **Calcolo Integrale Definito** (simbolico se possibile, altrimenti numerico)")
    try:
        # Simbolico
        F_approx = f_primitive.subs(t, x1) - f_primitive.subs(t, x0)
        F_approx = sp.simplify(F_approx)
        st.latex(fr"\int_{{{x0}}}^{{{x1}}} {expr_latex} \, d{variable} = " + sp.latex(F_approx))
    except Exception:
        st.info("Valore simbolico non disponibile")

    try:
        # Numerico (stabile anche per casi non integrabili simbolicamente)
        f_num = sp.lambdify(t, f_expr, modules=['numpy'])
        num_val, err_val = quad(f_num, x0, x1, limit=100)
        st.success(f"Valore numerico approssimato: **{num_val:.6f}** ± {err_val:.2g}")      
    except Exception as e:
        st.warning(f"Errore di integrazione numerica: {e}")

    # GRAFICO
    st.markdown("#### **Grafico integrale F(x)**")
    xs = np.linspace(xmin, xmax, 400)
    ys = []
    for x_ in xs:
        try:
            yval, _ = quad(f_lambd, x0, x_, limit=50)
        except:
            yval = np.nan
        ys.append(yval)
    ys = np.array(ys)
    fig, ax = plt.subplots(figsize=(10, 5))
    mask = ~np.isnan(ys)
    ax.plot(xs[mask], ys[mask], linewidth=2, label=f"F({variable})")
    ax.fill_between(xs[mask], ys[mask], 0, where=(ys[mask]>0), color='green', alpha=0.10)
    ax.fill_between(xs[mask], ys[mask], 0, where=(ys[mask]<0), color='red', alpha=0.08)
    # Area selezionata
    area_mask = (xs >= min(x0, x1)) & (xs<=max(x0, x1)) & mask
    ax.fill_between(xs[area_mask], ys[area_mask], 0, color='gold', alpha=0.25, label=f'Area tra x0 e x1')
    ax.axhline(0, color='k', lw=0.8)
    ax.axvline(0, color='k', lw=0.8)
    ax.legend()
    ax.grid(True, linestyle=':')
    st.pyplot(fig)

    plot_bytes = download_plot(fig)
    st.download_button(label="📥 Scarica grafico PNG", data=plot_bytes, file_name="integrale.png", mime="image/png")

# Modalità Allenamento/Quiz
with st.expander("🧑‍🎓 Modalità Allenamento/Quiz (Prototipo)"):
    st.write("Ottieni una formula random e prova a calcolarne l'integrale!")
    if st.button("Nuovo esercizio casuale"):
        import random
        code = random.choice(DEFAULT_FUNCS)[1]
        st.session_state['func_input'] = code
        st.session_state['var_input'] = "x"
        st.session_state['x0_input'] = 1.0
        st.session_state['x1_input'] = 2.0
        st.rerun()

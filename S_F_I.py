import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
from scipy.integrate import quad
from PIL import Image
import re
import warnings
import io

# OCR locale
try:
    from pix2tex.cli import LatexOCR
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False

st.set_page_config(page_title="Studio Integrale Local OCR+", layout="wide")
warnings.filterwarnings('ignore')

# --- Funzione migliorata per parsing LaTeX ---
def latex_to_python_manual(latex_str):
    s = latex_str

    # Estrazione variabile usata nell'integrale (\, lettere, ...)
    var_match = re.search(r'd([a-zA-Z])', s)
    variable = var_match.group(1) if var_match else 't'

    # Estrazione limiti integrale
    start_point, end_point = 0.0, 'x'
    try:
        match_limits = re.search(r"\\int_\{([^\}]+)\}\^\{([^\}]+)\}", s)
        if match_limits:
            lim0, lim1 = match_limits.group(1), match_limits.group(2)
            if 'pi' in lim0: start_point = np.pi
            elif 'infty' in lim0: start_point = 100
            else: start_point = float(lim0) if re.match(r"[-\d\.]+", lim0) else lim0
            if 'pi' in lim1: end_point = np.pi
            elif 'infty' in lim1: end_point = 100
            else: end_point = lim1
    except:
        pass

    # Elimina \int_...^{...}
    s = re.sub(r"\\int_[^{]*\{[^\}]+\}\^\{[^\}]+\}", "", s)   # caso \int_{..}^{..}
    s = re.sub(r"\\int_[^\^]+\^\{[^\}]+\}", "", s)            # altro caso simile
    s = re.sub(r"\\int", "", s)
    s = re.sub(r"d"+variable+"$", "", s).strip()

    while "\\frac" in s:
        s = re.sub(r"\\frac\{(.+?)\}\{(.+?)\}", r"(\1)/(\2)", s)
    # Funzioni
    s = s.replace(r"\sqrt", "sqrt")
    s = s.replace(r"\ln", "log")
    s = s.replace(r"\log", "log")
    s = s.replace(r"e^", "exp")
    s = s.replace(r"\sin", "sin").replace(r"\cos", "cos").replace(r"\tan", "tan")
    s = s.replace("^", "**")
    s = s.replace("{", "(").replace("}", ")")
    s = s.replace("\\", "")

    return s.strip(), variable, start_point, end_point

# --- CLASSE ANALYZER migliorata ---
class IntegralAnalyzer:
    def __init__(self, func_str, variable='t'):
        self.variable_name = variable
        self.t = sp.symbols(variable)
        self.func_str = func_str
        self.error = None
        # Symbolic
        try:
            self.f_expr = sp.sympify(func_str)
            self.f_numeric = sp.lambdify(self.t, self.f_expr, modules=['numpy'])
            self.f_deriv = sp.simplify(sp.diff(self.f_expr, self.t))
            self.f_primitive = sp.simplify(sp.integrate(self.f_expr, self.t))
        except Exception as e:
            self.error = str(e)

    def compute_numeric_integral(self, a, b):
        try:
            val, err = quad(self.f_numeric, a, b, limit=100)
            return val, err
        except Exception as exc:
            return None, str(exc)

    def compute_data(self, x0, x_range):
        xs = np.linspace(x_range[0], x_range[1], 400)
        ys = []
        for x in xs:
            try:
                val, _ = quad(self.f_numeric, x0, x, limit=50)
                ys.append(val)
            except:
                ys.append(np.nan)
        return xs, np.array(ys)

# --- CACHING OCR model ---
@st.cache_resource
def load_ocr_model():
    if OCR_AVAILABLE:
        return LatexOCR()
    return None

ocr_model = load_ocr_model()

# --- FUNZIONI ESPORTAZIONE ---
def download_plot(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    buf.seek(0)
    return buf

def latex_to_pdf(latex_code, filename="formula.pdf"):
    import matplotlib.backends.backend_pdf as pdf_backend
    fig, ax = plt.subplots(figsize=(5, 1))
    ax.axis('off')
    ax.text(0.5, 0.5, f"${latex_code}$", fontsize=20, ha='center', va='center')
    pdf = pdf_backend.PdfPages(filename)
    pdf.savefig(fig)
    pdf.close()
    plt.close(fig)
    return filename

# ------------------- UI -------------------
st.title("📚 Studio Integrale - OCR & Calcolo Symbolico")
st.markdown("""
App per studenti di ingegneria: estrai formule, analizza e visualizza grafici degli integrali.  
Funzioni avanzate: derivata, primitiva simbolica, calcolo definito, download risultati, tutorial!
""")

if 'func_input' not in st.session_state: st.session_state['func_input'] = "log(t)/(t*(t-1))"
if 'var_input' not in st.session_state: st.session_state['var_input'] = "t"
if 'x0_input' not in st.session_state: st.session_state['x0_input'] = 1.0
if 'x1_input' not in st.session_state: st.session_state['x1_input'] = 2.0

with st.sidebar:
    st.header("📸 1. Scanner Formula/OCR")
    if not OCR_AVAILABLE:
        st.error("Libreria 'pix2tex' non trovata. Installa con `pip install pix2tex`")
    else:
        uploaded_file = st.file_uploader("Carica immagine formula (possibilmente in stampa)", type=["png", "jpg", "jpeg"])
        if uploaded_file is not None and st.button("Analizza Immagine"):
            with st.spinner("Decodifica LaTeX in corso..."):
                try:
                    img = Image.open(uploaded_file)
                    raw_latex = ocr_model(img)
                    st.caption(f"Letto: `${raw_latex}$`")
                    py_func, var, py_x0, py_x1 = latex_to_python_manual(raw_latex)
                    st.session_state['func_input'] = py_func
                    st.session_state['var_input'] = var
                    st.session_state['x0_input'] = py_x0 if isinstance(py_x0, (int,float)) else 0.0
                    st.session_state['x1_input'] = py_x1 if isinstance(py_x1, (int,float)) else 2.0
                    st.success("Conversione riuscita!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Errore lettura: {e}")

    st.divider()
    st.header("✏️ 2. Formula e Parametri")
    func_input = st.text_input("Funzione (es: log(t)/(t*(t-1)))", value=st.session_state['func_input'])
    var_input = st.text_input("Variabile (t/x/...) :", value=st.session_state['var_input'])
    x0 = st.number_input("Punto di partenza $x_0$", value=float(st.session_state['x0_input']))
    x1 = st.number_input("Punto di arrivo $x_1$ (per calcolo definito):", value=float(st.session_state['x1_input']))
    col1, col2 = st.columns(2)
    xmin = col1.number_input("Min. grafico", value=-1.0)
    xmax = col2.number_input("Max. grafico", value=5.0)

    st.divider()
    st.header("ℹ️ Tutorial & Aiuto rapido")
    st.markdown("""
- **OCR**: scegli un'immagine con una formula LaTeX leggibile.
- **Correggi eventuali errori**: a volte l'OCR sbaglia parentesi! Ricontrolla la funzione e la variabile.
- **Puoi modificare formula, variabile e limiti manualmente.**
- **Cosa puoi fare**:
    - Vedere: primitiva simbolica, derivata della funzione, integrale numerico tra due punti
    - Visualizzare aree colorate tra due punti scelti
    - Scaricare grafici/risultati
- **Se ottieni errori**: assicurati che la sintassi sia giusta, che la variabile usata sia la stessa ovunque, e che non ci siano lettere strane (es: “𝑡” vs “t”).
- **Esempio formula**: `\\int_{1}^{x} \\frac{\\ln(t)}{t\\,(t-1)}dt`
- Powered by Python/SymPy/Streamlit
""")

# ------------- SEZIONE PRINCIPALE -------------
st.header("🧑‍🔬 Analisi formula/Integrale")
if func_input:
    analyzer = IntegralAnalyzer(func_input, variable=var_input)
    if analyzer.error:
        st.error(f"Errore sintassi: {analyzer.error}")
        st.info("Correggi parentesi o controlla nome della variabile!")
        st.stop()
    else:
        colF1, colF2, colF3 = st.columns(3)
        with colF1:
            st.latex(r"f("+analyzer.variable_name+") = " + sp.latex(analyzer.f_expr))
        with colF2:
            st.latex(r"f'("+analyzer.variable_name+") = " + sp.latex(analyzer.f_deriv))
        with colF3:
            st.latex(r"\int " + sp.latex(analyzer.f_expr) + " d"+analyzer.variable_name +
                     r" = " + sp.latex(analyzer.f_primitive) + " + C")

        # Mostra primitiva calcolata anche con pretty-print (step simbolico)
        with st.expander("Passaggi simbolici (SymPy)", expanded=False):
            st.write("**Forma step-by-step (primitiva):**")
            st.code(sp.pretty(analyzer.f_primitive), language='python')

        # Calcolo integrale definito tra x0 e x1
        st.subheader("Risultato Integrale Definito")
        try:
            # Valore simbolico se possibile
            F_sym = analyzer.f_primitive
            F_num_x1 = F_sym.subs(analyzer.t, x1)
            F_num_x0 = F_sym.subs(analyzer.t, x0)
            sym_res = sp.simplify(F_num_x1 - F_num_x0)
            st.latex(r"\int_{"+str(x0)+r"}^{"+str(x1)+r"} " + sp.latex(analyzer.f_expr) + " d"+analyzer.variable_name +
                    r" = " + sp.latex(sym_res))
        except Exception:
            st.write("Non è stato possibile calcolare il valore simbolico.")

        # Calcolo numerico
        num_val, err_val = analyzer.compute_numeric_integral(x0, x1)
        if num_val is not None:
            st.success(f"Valore numerico approssimato: **{num_val:.6f}**  ± {err_val:.2g}")

        # --- GRAFICO E AREAS (funzionalità avanzate)
        xs, ys = analyzer.compute_data(x0, (xmin, xmax))
        fig, ax = plt.subplots(figsize=(10, 5))
        mask = ~np.isnan(ys)
        ax.plot(xs[mask], ys[mask], linewidth=2, label="Primitiva $F(x)$")
        area_mask = (xs >= min(x0, x1)) & (xs <= max(x0, x1)) & mask
        ax.fill_between(xs[area_mask], ys[area_mask], 0, color='gold', alpha=0.3, label=f'Area $x_0 \\to x_1$')
        ax.fill_between(xs[mask], ys[mask], 0, where=(ys[mask]>0), color='green', alpha=0.08)
        ax.fill_between(xs[mask], ys[mask], 0, where=(ys[mask]<0), color='red', alpha=0.08)
        ax.axhline(0, color='k', linewidth=0.7); ax.axvline(0, color='k', linewidth=0.7)
        ax.grid(True, linestyle=':')
        ax.legend()
        st.pyplot(fig)

        # Download pulsanti
        plot_bytes = download_plot(fig)
        st.download_button(label="📥 Scarica grafico PNG", data=plot_bytes,
                           file_name="integrale.png", mime="image/png")
        st.download_button(label="📥 Scarica formula LaTeX (PDF)", 
                           data=open(latex_to_pdf(sp.latex(analyzer.f_expr)), "rb").read(),
                           file_name="funzione.pdf", mime="application/pdf")

# (Opzionale) Modalità Allenamento/Quiz
with st.expander("🧑‍🎓 Modalità Allenamento/Quiz (Prototipo)"):
    st.write("""
Clicca per ottenere un integrale casuale ed esercitati!
""")
    if st.button("Nuovo esercizio casuale"):
        examples = [
            ("exp(-t**2)", "t", 0, 2),
            ("sin(t)/t", "t", 1, 5),
            ("cos(t)*exp(t)", "t", 0, 1),
            ("log(t+1)", "t", 0, 2),
        ]
        import random
        f, var, a, b = random.choice(examples)
        st.session_state['func_input'] = f
        st.session_state['var_input'] = var
        st.session_state['x0_input'] = a
        st.session_state['x1_input'] = b
        st.rerun()

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
from scipy.integrate import quad
from PIL import Image
import re
import warnings

# Importazione OCR Locale (con gestione errore se non installato)
try:
    from pix2tex.cli import LatexOCR
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False

st.set_page_config(page_title="Studio Integrale Local OCR", layout="wide")
warnings.filterwarnings('ignore')

# --- MOTORE DI PARSING (Da LaTeX a Python) ---
def latex_to_python_manual(latex_str):
    """
    Traduttore artigianale da LaTeX a sintassi SymPy/Python.
    Non è perfetto ma copre il 90% dei casi da libro di testo.
    """
    s = latex_str
    
    # 1. Estrazione Estremo Inferiore (cerca pattern come \int_{-1} o \int_{-1}^x)
    # Regex: cerca \int seguito da _ e cattura il contenuto tra graffe {} o singolo char
    start_point = 0.0
    try:
        match_limit = re.search(r"\\int_\{?([^\}^]+)\}?\^", s)
        if match_limit:
            limit_str = match_limit.group(1)
            # Gestione pi greco o infiniti nel limite
            if 'pi' in limit_str: start_point = np.pi
            elif 'infty' in limit_str: start_point = 100 # euristico
            else: start_point = float(limit_str)
    except:
        pass # Mantiene default 0.0 se fallisce

    # 2. Pulizia per isolare la funzione integranda
    # Rimuove \int...dt e lascia solo il corpo
    s = re.sub(r"\\int.*?\^\{?x\}?", "", s) # Via l'integrale e gli estremi
    s = re.sub(r"d[t|x]\s*$", "", s) # Via il differenziale finale
    
    # 3. Traduzione Sintassi Matematica
    # Frazioni: \frac{A}{B} -> (A)/(B)
    # Nota: Questo loop gestisce frazioni annidate semplici
    while "\\frac" in s:
        s = re.sub(r"\\frac\{(.+?)\}\{(.+?)\}", r"(\1)/(\2)", s)
    
    # Radici
    s = s.replace(r"\sqrt", "sqrt")
    # Logaritmi
    s = s.replace(r"\ln", "log")
    s = s.replace(r"\log", "log")
    # Esponenziali e Trigonometria
    s = s.replace(r"e^", "exp")
    s = s.replace(r"\sin", "sin").replace(r"\cos", "cos").replace(r"\tan", "tan")
    # Potenze (LaTeX usa ^, Python usa **)
    s = s.replace("^", "**")
    # Pulizia parentesi LaTeX rimaste (es. {t})
    s = s.replace("{", "(").replace("}", ")")
    # Backslash rimasti
    s = s.replace("\\", "")
    
    return s.strip(), start_point

# --- CLASSE ANALYZER (Invariata) ---
class IntegralAnalyzer:
    def __init__(self, func_str, variable='t'):
        self.t = sp.symbols(variable)
        self.func_str = func_str
        self.error = None
        try:
            self.f_expr = sp.sympify(func_str)
            self.f_numeric = sp.lambdify(self.t, self.f_expr, modules=['numpy'])
        except Exception as e:
            self.error = str(e)

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

# --- CACHING DEL MODELLO OCR ---
@st.cache_resource
def load_ocr_model():
    if OCR_AVAILABLE:
        return LatexOCR()
    return None

ocr_model = load_ocr_model()

# --- UI ---
st.title("📚 Studio Integrale - Local Textbook Reader")
st.markdown("Nessuna API Google richiesta. Usa modelli AI locali.")

if 'func_input' not in st.session_state: st.session_state['func_input'] = "log(t)/(t*(t-1))"
if 'x0_input' not in st.session_state: st.session_state['x0_input'] = 1.0

with st.sidebar:
    st.header("📸 Scanner Formule")
    
    if not OCR_AVAILABLE:
        st.error("Libreria 'pix2tex' non trovata. Installala con pip.")
    else:
        uploaded_file = st.file_uploader("Carica immagine formula (testo stampato)", type=["png", "jpg"])
        
        if uploaded_file is not None and st.button("Analizza Immagine"):
            with st.spinner("Decodifica LaTeX in corso..."):
                try:
                    # Caricamento immagine per PIL
                    img = Image.open(uploaded_file)
                    
                    # 1. OCR: Immagine -> LaTeX string
                    raw_latex = ocr_model(img)
                    st.caption(f"Letto: `${raw_latex}$`")
                    
                    # 2. Parsing: LaTeX -> Python
                    py_func, py_x0 = latex_to_python_manual(raw_latex)
                    
                    st.success("Conversione riuscita!")
                    st.session_state['func_input'] = py_func
                    st.session_state['x0_input'] = py_x0
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"Errore lettura: {e}")

    st.divider()
    func_input = st.text_input("Funzione f(t):", value=st.session_state['func_input'])
    x0 = st.number_input("x0:", value=float(st.session_state['x0_input']))
    
    col1, col2 = st.columns(2)
    xmin = col1.number_input("Min", -1.0)
    xmax = col2.number_input("Max", 5.0)

# --- CORE LOGIC ---
if func_input:
    analyzer = IntegralAnalyzer(func_input)
    if analyzer.error:
        st.error(f"Errore sintassi: {analyzer.error}")
        st.info("Suggerimento: L'OCR potrebbe aver lasciato parentesi strane. Correggi manualmente.")
    else:
        st.latex(r"F(x) = \int_{" + str(x0) + r"}^{x} " + sp.latex(analyzer.f_expr) + " dt")
        
        xs, ys = analyzer.compute_data(x0, (xmin, xmax))
        
        fig, ax = plt.subplots(figsize=(10, 5))
        mask = ~np.isnan(ys)
        ax.plot(xs[mask], ys[mask], linewidth=2)
        ax.fill_between(xs[mask], ys[mask], 0, where=(ys[mask]>0), color='green', alpha=0.1)
        ax.fill_between(xs[mask], ys[mask], 0, where=(ys[mask]<0), color='red', alpha=0.1)
        ax.axhline(0, color='k'); ax.axvline(0, color='k')
        ax.grid(True, linestyle=':')
        st.pyplot(fig)
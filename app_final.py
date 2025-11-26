# ============================================
# 📦 IMPORTACIONES Y CONFIGURACIÓN GENERAL
# ============================================
import streamlit as st

# SET PAGE CONFIG: SIEMPRE PRIMERO
st.set_page_config(
    page_title="DocuSmart",
    page_icon="📘",
    layout="centered"   
)

import os
import io
import re
import json
import tempfile
import unicodedata
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from pdf2image import convert_from_bytes, convert_from_path
import google.generativeai as genai
from google.cloud import vision

Image.MAX_IMAGE_PIXELS = None

# GOOGLE VISION CONFIG
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "clave.json"
cliente_vision = vision.ImageAnnotatorClient()

# GEMINI CONFIG
genai.configure(api_key="API_KEY")

# ============================================
# 🧰 FUNCIONES UTILITARIAS
# ============================================

def normalizar(texto):
    """Normaliza acentos y convierte a minúsculas"""
    return unicodedata.normalize("NFKD", texto).encode("ascii", "ignore").decode().lower()

def es_vacio(valor):
    """Verifica si un valor es vacío, nulo o texto irrelevante"""
    return valor is None or (isinstance(valor, str) and valor.strip().lower() in ["", "n/a", "null", "none"])

def decodificar_json_limpio(texto):
    """Intenta limpiar y decodificar un texto como JSON"""
    if not texto or not isinstance(texto, str):
        return {}
    texto = texto.strip().strip("`")
    texto = re.sub(r"^json", "", texto, flags=re.IGNORECASE).strip()
    try:
        return json.loads(texto)
    except:
        try:
            texto = texto.replace("'", '"')
            return json.loads(texto)
        except:
            return {}

def mostrar_vista_preliminar(pdf_bytes, nombre_archivo):
    """
    Muestra imagen original y preprocesada del PDF subido,
    y almacena los bytes en el session_state.
    """
    paginas = convert_from_bytes(pdf_bytes, dpi=600, first_page=1, last_page=1)
    imagen_original = paginas[0]
    st.image(imagen_original, caption="🖼️ Imagen original", width="stretch")

    imagen_preprocesada = preprocess_for_vision(imagen_original)
    st.image(imagen_preprocesada, caption="🧪 Imagen preprocesada", width="stretch")

    # Guardar para procesamiento posterior
    st.session_state[f"pdf_bytes_{nombre_archivo}"] = pdf_bytes

def procesar_y_mostrar_resultados(pdf_bytes, archivo_nombre, regiones, prompts, modelo):
    if st.button(f"📝 Transcribir `{archivo_nombre}`"):
        df = procesar_documento(
            pdf_bytes=pdf_bytes,
            regiones=regiones,
            prompts=prompts,
            modelo=modelo,
            aplicar_mejoramiento_extra=True
        )
        st.dataframe(df)

        output = exportar_excel(df)
        st.download_button(
            label=f"📥 Descargar resultados de {archivo_nombre}",
            data=output,
            file_name=f"resultados_{archivo_nombre}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

def extraer_matricula_desde_nombre(nombre_archivo):
    """
    Extrae del nombre del archivo la porción válida como matrícula:
    letras, números y guion, hasta antes de paréntesis, espacios u otros caracteres.
    Ejemplo:
        '7S-39280 (l29-031).pdf' → '7S-39280'
    """
    base = os.path.splitext(nombre_archivo)[0]  # quitar .pdf
    base = base.strip()

    # Cortar antes de cualquier cosa que no sea alfanumérico o guion
    corte = re.split(r"[^\w\-]", base)[0]

    return corte.upper()  # para estandarizar

def corregir_matricula_con_nombre(matricula_detectada, nombre_archivo):
    """
    Usa SIEMPRE la matrícula derivada del nombre del archivo.
    La matrícula OCR solo se usa para determinar la longitud de coincidencia.
    """

    mat_archivo = extraer_matricula_desde_nombre(nombre_archivo)

    if not mat_archivo:
        return matricula_detectada  # fallback: usar OCR

    if matricula_detectada:
        # Emparejar longitud con la detectada
        longitud = len(matricula_detectada)
        return mat_archivo[:longitud]

    # Si OCR está vacío, usar matrícula derivada del archivo
    return mat_archivo

# ============================================
# 🧪 PREPROCESAMIENTO DE IMAGEN PARA OCR
# ============================================

def preprocess_for_vision(pil_img):
    """
    Preprocesa la imagen para mejorar el contraste y mejorar la extraccion del OCR.
    """
    img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enh = clahe.apply(gray)
    return Image.fromarray(enh)

# ============================================
# ✂️ RECORTE DE CAMPOS DE INTERÉS
# ============================================

def recortar_campos(img_pil, regiones):
    """
    Recorta regiones definidas de una imagen PIL.
    - img_pil: imagen ya preprocesada
    - regiones: diccionario con claves de campo y valores (x1, y1, x2, y2)
    """
    return {
        campo: img_pil.crop((x1, y1, x2, y2))
        for campo, (x1, y1, x2, y2) in regiones.items()
    }

# ============================================
# 🔧 MEJORAMIENTO SECUNDARIO (Sauvola)
# ============================================

def aplicar_segundo_mejoramiento(pil_img):
    """
    Aplica un mejoramiento local con umbral Sauvola sobre una imagen recortada.
    Muy útil en textos con fondo gris o mala iluminación.
    """
    img_cv = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    gris = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)

    # Umbral local adaptativo Sauvola
    window_size = 25
    sauvola = threshold_sauvola(gris, window_size=window_size)
    binaria = img_as_ubyte(gris > sauvola)

    # Upscaling y enfoque
    escalada = cv2.resize(binaria, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharpened = cv2.filter2D(escalada, -1, kernel)

    return Image.fromarray(sharpened)

# ============================================
# 🔍 OCR CON VISION
# ============================================

def vision_ocr(pil_img):
    """
    Extrae de una porcion de texto especifica. 
    """
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    content = buf.getvalue()

    image = vision.Image(content=content)
    response = cliente_vision.text_detection(image=image)

    if not response.text_annotations:
        return ""
    return response.text_annotations[0].description

# ============================================
# 🤖 CONSULTA AL LLM (GEMINI)
# ============================================
        
def normalizar_con_gemini(ocr_campos):
    prompt = f"""
      Convertí estrictamente el siguiente texto OCR en un JSON ***válido***.
      El texto es una transcripcion de matriculas en formato PDF del registro de la propiedad de Santiago del Estero, Argentina.
      IMPORTANTE:
      - NO agregues explicaciones ni texto afuera del JSON
      - NO agregues comentarios
      - NO uses comillas incorrectas
      - NO inventes datos
      - Si falta un dato, dejá el campo vacío ("")
      - Devolvé SOLO el JSON válido, nada más.
      - Extrae los datos del titular de dominio: nombre completo, el apellido debe existir

      Estructura obligatoria:
      {{
        "matricula": "",
        "ubicacion": {{
          "departamento": "",
          "designacion": ""
        }},
        "titulares": [
          {{
            "nombre": "",
            "dni": "",
            "cuil": ""
          }}
        ],
        "plano": "",
        "medidas_linderos": {{
          "descripcion": "",
          "medidas": [
            {{
              "tramo": "",
              "longitud": ""
            }}
          ],
          "linderos": {{
            "sudeste": "",
            "sudoeste": "",
            "noroeste": "",
            "noreste": ""
          }}
        }},
        "superficie": "",
        "observaciones": ""
      }}

      Texto OCR crudo:
      {ocr_campos}
      """


    model = genai.GenerativeModel("gemini-2.5-pro")
    resp = model.generate_content(prompt)
    return resp.text

# ============================================
# 📊 LIMPIEZA Y VALIDACION JSON
# ============================================

def limpiar_bloque_json(texto):
    """Quita ```json y markdown basura."""
    if texto is None:
        return ""
    t = re.sub(r"```json", "", texto, flags=re.IGNORECASE)
    t = re.sub(r"```", "", t)
    lineas = t.strip().splitlines()
    if lineas and lineas[0].strip().lower() == "json":
        lineas = lineas[1:]
    return "\n".join(lineas).strip()

def validar_json(texto, nombre_pdf):
    limpio = limpiar_bloque_json(texto)
    try:
        return json.loads(limpio)
    except:
        st.error(f"❌ JSON inválido en {nombre_pdf}")
        st.code(limpio)
        raise

# ============================================
# 📐 NORMALIZADOR DE PLANO
# ============================================

def normalizar_plano(plano_raw):
    """
    Normalizacion del campo 'plano' al formato [Tºxxx Fºxx/xxx].
    """
    if not plano_raw or not isinstance(plano_raw, str):
        return ""

    # Extraer todos los números del texto
    nums = re.findall(r"\d+", plano_raw)
    
    # Caso esperado: al menos 2 bloques
    if len(nums) >= 2:
        
        # --- TOMO ---
        tomo_raw = nums[0]

        # Casos comunes:
        # T9508 → tomo = 508
        # T058 → tomo = 58 (si OCR mete un 0 basura)
        # Regla: si tiene 4 dígitos y empieza con 9 → es OCR basura → quitar primer dígito
        if len(tomo_raw) == 4 and tomo_raw.startswith("9"):
            tomo = tomo_raw[1:]   # "9508" → "508"
        elif len(tomo_raw) == 3:
            tomo = tomo_raw       # "508"
        else:
            tomo = tomo_raw       # fallback
        
        # --- FOLIO ---
        folio_raw = "".join(nums[1:])

        # Debe tener al menos 4 dígitos para separar en folio + fracción
        if len(folio_raw) >= 4:
            folio_entero = folio_raw[:-3]
            folio_decimal = folio_raw[-3:]
            folio = f"{folio_entero}/{folio_decimal}"
        else:
            folio = folio_raw

        return f"Tº{tomo} Fº{folio}"

    # fallback si no coincide con nada
    return plano_raw


def aplanar_json(j, nombre_archivo):
    ubic = j.get("ubicacion", {})
    titulares = j.get("titulares", [])
    medidas = j.get("medidas_linderos", {})
    lista_medidas = medidas.get("medidas", [])
    lind = medidas.get("linderos", {})

    # MATRÍCULA OCR
    matricula_detectada = j.get("matricula", "")

    # MATRÍCULA CORREGIDA POR NOMBRE DEL ARCHIVO
    matricula_final = corregir_matricula_con_nombre(matricula_detectada, nombre_archivo)

    # ========================================
    # 🟦 FORMAR CAMPO "mide_y_linda"
    # ========================================

    descripcion = medidas.get("descripcion", "").lower()

    # Medidas → lista formateada
    medidas_txt = ""
    if lista_medidas:
        medidas_txt = "\n".join(
            [f" - {m.get('tramo','').lower()}: {m.get('longitud','').lower()}" 
             for m in lista_medidas]
        )

    # Linderos en un solo bloque ↓↓
    linderos_txt = f"""
 - sudeste: {lind.get("sudeste","").lower()}
 - sudoeste: {lind.get("sudoeste","").lower()}
 - noroeste: {lind.get("noroeste","").lower()}
 - noreste: {lind.get("noreste","").lower()}
 """

    # Unificar TODO
    mide_y_linda = f"""descripción: {descripcion}

medidas:
{medidas_txt}

linderos:
{linderos_txt}
"""
    # Devuelve solo la fila(solo campos necesarios)
    return {
        "nombre_archivo": nombre_archivo.lower(),
        "matricula": matricula_final.lower(),
        "departamento": ubic.get("departamento", "").lower(),
        "designacion": ubic.get("designacion", "").lower(),
        "titulares_nombres": "; ".join(t.get("nombre", "").lower() for t in titulares),
        "titulares_dnis": "; ".join(t.get("dni", "").lower() for t in titulares),
        "titulares_cuils": "; ".join(t.get("cuil", "").lower() for t in titulares),
        "plano": j.get("plano", "").lower(),
        "plano_normalizado": normalizar_plano(j.get("plano", "")).lower(),
        "superficie": j.get("superficie", "").lower(),
        "mide_y_linda": mide_y_linda.strip().lower()
    }

# ============================================
# 📊 CONVERSIÓN A EXCEL
# ============================================

def convertir_a_excel(filas: list[dict]):
    """
    Convierte una lista de diccionarios en un DataFrame.
    """
    df = pd.DataFrame(filas)
    return df

def exportar_excel(df):
    output = io.BytesIO()
    df.to_excel(output, index=False)
    output.seek(0)
    return output

# ============================================
# 🚀 PIPELINE PRINCIPAL DE PROCESAMIENTO
# ============================================

def procesar_documento(pdf_bytes, regiones, nombre_archivo):
    """
    Pipeline completo: carga, mejora, recorte, análisis con Gemini por imagen, exportación.
    """
    paginas = convert_from_bytes(pdf_bytes, dpi=600, first_page=1, last_page=1)
    imagen = paginas[0]

    img_proc = preprocess_for_vision(imagen)
    recortes = recortar_campos(img_proc, regiones)

    ocr_campos = {campo: vision_ocr(im) for campo, im in recortes.items()}

    json_texto = normalizar_con_gemini(ocr_campos)
    json_validado = validar_json(json_texto, "documento")

    fila = aplanar_json(json_validado, nombre_archivo)
    return pd.DataFrame([fila])

# ============================================
# 🔐 CONTROL DE ACCESO - LOGIN STREAMLIT
# ============================================

USUARIOS = {
    "admin": "admin123",
    "leandro": "123"
}

def login():
    # --- CSS para centrar y limitar el ancho del login ---
    st.markdown("""
        <style>
            .login-box {
                max-width: 600px;
                margin: auto;
                padding: 20px;
            }
        </style>
    """, unsafe_allow_html=True)

    with st.container():
        st.markdown("<div class='login-box'>", unsafe_allow_html=True)

        st.markdown("""
            <div style='text-align:center; margin-top:30px;'>
                <h1 style='font-size:58px; margin-bottom:-10px;'>📘 DocuSmart</h1>
                <h3 style='color:#CCCCCC; font-weight:300;'>
                    🔐 Iniciar sesión
                </h3>
            </div>
        """, unsafe_allow_html=True)

        usuario = st.text_input("Usuario")
        clave = st.text_input("Contraseña", type="password")

        if st.button("Ingresar"):
            if USUARIOS.get(usuario) == clave:
                st.session_state["logueado"] = True
                st.session_state["usuario"] = usuario
                st.rerun()
            else:
                st.error("❌ Usuario o contraseña incorrectos")

        st.markdown("</div>", unsafe_allow_html=True)


# Validación
if not st.session_state.get("logueado"):
    login()
    st.stop()

# Definición de regiones y prompts
regiones = {
    "matricula": (900, 900, 2000, 1600),
    "ubicacion": (2000, 870, 5900, 1600),
    "titular":   (900, 3300, 4200, 5000),
    "plano":     (5870, 1190, 7470, 1500),
    "medidas":   (900, 1540, 7300, 3300),
}

# ============================================
# 🖥️ INTERFAZ STREAMLIT — PROCESAR TODO JUNTO (RESTABLECIDA)
# ============================================

# --- SIDEBAR ---
with st.sidebar:

    st.markdown("## 👤 Usuario")
    st.success(f"Sesión iniciada como **{st.session_state['usuario']}**")

    st.markdown("---")
    st.markdown("### 📎 Subí documentos PDF")

    uploaded_files = st.file_uploader(
        label="",
        type="pdf",
        accept_multiple_files=True,
        label_visibility="collapsed"
    )

    st.markdown("---")

    if st.button("🚪 Cerrar sesión", width="stretch"):
        st.session_state.clear()
        st.rerun()


# --- ENCABEZADO ---
st.markdown(f"""
<div style='text-align:center; margin-top:10px;'>
    <h1 style='font-size:58px; margin-bottom:-10px;'>📘 DocuSmart</h1>
    <h3 style='color:#CCCCCC; font-weight:300;'>
        Transcriptor de Matrículas – Registro de la Propiedad
    </h3>
</div>
""", unsafe_allow_html=True)


# ============================================
# 📄 ARCHIVOS SUBIDOS (CON PREVIEW EN EXPANDERS)
# ============================================
if uploaded_files:

    st.markdown("## 📂 Archivos cargados")

    previews = {}

    for archivo in uploaded_files:

        pdf_bytes = archivo.read()

        paginas = convert_from_bytes(pdf_bytes, dpi=600, first_page=1, last_page=1)
        imagen_original = paginas[0]
        imagen_pre = preprocess_for_vision(imagen_original)

        previews[archivo.name] = (pdf_bytes, imagen_original, imagen_pre)

        with st.expander(f"📄 {archivo.name}", expanded=False):
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("#### Imagen original")
                st.image(imagen_original, width="stretch")

            with col2:
                st.markdown("#### Imagen preprocesada")
                st.image(imagen_pre, width="stretch")

    st.markdown("---")

    # ================================
    # 🔵 BOTÓN ÚNICO PARA PROCESAR TODO
    # ================================
    
    st.markdown("""
    <style>

    .stButton>button {
        border-radius: 999px !important;
        padding: 0.6rem 1.8rem !important;
        font-size: 1rem !important;
        font-weight: 600 !important;
        border: none !important;
        cursor: pointer !important;
        transition: 0.2s ease-in-out !important;
    }

    /* Botón principal: transcribir */
    #transcribir_btn button {
        background: linear-gradient(135deg, #2563eb, #22c55e) !important;
        color: white !important;
        box-shadow: 0 8px 20px rgba(0,0,0,0.4) !important;
    }
    #transcribir_btn button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 12px 26px rgba(0,0,0,0.55) !important;
    }

    /* Botón de descarga */
    #descargar_btn button {
        background: #0f172a !important;
        color: #e5e7eb !important;
        border: 1px solid #1f2937 !important;
        box-shadow: 0 6px 18px rgba(0,0,0,0.35) !important;
    }
    #descargar_btn button:hover {
        background: #1e293b !important;
        transform: translateY(-2px) !important;
    }

    .center {
        width: 100%;
        display: flex;
        justify-content: center;
        margin: 20px 0;
    }

    </style>
    """, unsafe_allow_html=True)

    # === Botón con clase CSS personalizada ===
    st.markdown('<div class="center" id="transcribir_btn">', unsafe_allow_html=True)
    procesar_todo = st.button("🚀 Transcribir todos los documentos", key="btn_transcribir")
    st.markdown('</div>', unsafe_allow_html=True)

    if procesar_todo:

        total = len(previews)
        progreso = st.progress(0, text="Iniciando...")

        resultados_globales = []

        for idx, (nombre, (pdf_bytes, img_raw, img_pre)) in enumerate(previews.items(), start=1):

            progreso.progress((idx - 1) / total, text=f"Procesando {nombre}…")

            df = procesar_documento(pdf_bytes, regiones, nombre)
            resultados_globales.append(df)

        progreso.progress(1.0, text="✔️ Procesamiento completado")

        st.success("✔️ Todos los documentos fueron procesados correctamente")

        df_final = pd.concat(resultados_globales, ignore_index=True)

        df_final = df_final.applymap(lambda x: x.lower() if isinstance(x, str) else x)

        st.markdown("## 📊 Resultados")
        st.dataframe(df_final, width="stretch")

        excel_bytes = exportar_excel(df_final)

        st.markdown('<div class="center" id="descargar_btn">', unsafe_allow_html=True)
        st.download_button(
            label="📥 Descargar Excel único",
            data=excel_bytes,
            file_name="resultados_completos.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key="btn_descargar"
        )
        st.markdown('</div>', unsafe_allow_html=True)


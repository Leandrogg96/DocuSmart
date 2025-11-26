# 📘 DocuSmart OCR  
### Transcriptor inteligente de Matrículas del Registro de la Propiedad – Santiago del Estero

![Python](https://img.shields.io/badge/Python-3.12-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red)
![Google Cloud](https://img.shields.io/badge/Google_Cloud-Vision-yellow)
![Gemini](https://img.shields.io/badge/Gemini-2.5_Pro-purple)
![Status](https://img.shields.io/badge/Build-Stable-brightgreen)

---

DocuSmart es una aplicación avanzada construida con **Python + Streamlit** que permite procesar documentos PDF del Registro de la Propiedad Inmueble para extraer información estructurada y normalizada mediante un pipeline híbrido:

- 📸 **Google Cloud Vision** para OCR
- 🤖 **Gemini 2.5 Pro** para reconstrucción semántica y JSON validado
- 🧠 **Preprocesamiento avanzado** para mejorar la calidad del OCR
- 📂 Procesamiento por lote (múltiples PDFs)
- 📄 Exportación automática a Excel con campos normalizados

Pensado para trabajar con **matrículas, planos, titulares, medidas y linderos**, obteniendo un Excel homogéneo y apto para uso administrativo.

---

## Características principales

### ✔️ OCR híbrido (Vision + Gemini)
El sistema combina extracción de texto con inteligencia semántica para reconstruir campos incluso en documentos deteriorados o escaneados.

### ✔️ Normalización legal
Incluye reglas exactas para:

- normalizar matrícula en base al nombre del archivo  
- corregir plano al formato **Tºxx Fºxx/xxx**  
- consolidar todos los linderos en el campo **mide_y_linda**

### ✔️ Previsualización de PDFs
Antes de procesar, el usuario puede ver:

- Imagen original  
- Imagen preprocesada  

### ✔️ Procesamiento masivo
Un solo botón procesa **todos** los PDFs cargados.

### ✔️ Exportación profesional a Excel
Exporta un archivo único con toda la información normalizada y en minúsculas.

---

## Flujo de procesamiento

1. **Carga de PDFs**
2. Conversión PDF → PNG
3. **Preprocesamiento** (CLAHE, sharpening)
4. **Recortes automáticos** por coordenadas
5. **OCR con Google Vision**
6. **Generación de JSON con campos clave** mediante Gemini
7. **Normalización de matrícula y planos**
8. Exportación a **Excel**

---

## Tecnologías utilizadas

- Python 3.12  
- Streamlit  
- Google Vision API  
- Google Gemini 2.5 Pro  
- OpenCV  
- scikit-image  
- pdf2image  
- Pandas
- Numpy

---

## Instalación

### 1. Clonar el repositorio
```bash
git clone https://github.com/tuusuario/DocuSmart.git
cd DocuSmart
```

## 2. Crear entorno virtual
```bash
python3 -m venv venv
source venv/bin/activate        # macOS / Linux
venv\Scripts\activate           # Windows
```

## 3. Instalar dependencias
```bash
pip install -r requirements.txt
```

## 4. Configurar Google Vision
```bash
export GOOGLE_APPLICATION_CREDENTIALS="clave.json"
```

## 5. Ejecutar la applicacion
```bash
streamlit run app_final.py
```

## 6. Abrir navegador
```bash
http://localhost:8501
```

---

## 📂 Estructura del proyecto
```bash
DocuSmart/
│── app_final.py
│── requirements.txt
│── .gitignore
│── clave.json           (NO subir al repo)
│── docs/                (capturas opcionales)
└── resultados/          (exportaciones Excel)
```

---

## 📜 Exportación a Excel – Campos incluidos

| Campo             | Descripción                                         |
|-------------------|-----------------------------------------------------|
| nombre_archivo    | Nombre del PDF original                             |
| matricula         | Matrícula corregida                                 |
| departamento      | Departamento                                        |
| designacion       | Designación catastral                               |
| titulares_nombres | Nombres                                             |
| titulares_dnis    | DNIs                                                |
| titulares_cuils   | CUILs                                               |
| plano             | Plano original                                      |
| plano_normalizado | Formato Tºxx Fºxx/xxx                               |
| superficie        | Superficie declarada                                |
| mide_y_linda      | **Todos los linderos + medidas consolidados**       |

---

## 🔐 Seguridad

⚠️ **Nunca subas `clave.json` al repositorio.**  
Asegurate de incluirlo en `.gitignore`.

---

## 🤝 Contribución

Pull requests y sugerencias son bienvenidas.  
Para reportar errores, abrí un issue.

---

## 📘 Licencia

Uso privado

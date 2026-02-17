import json
import random
import time
from datetime import datetime

import streamlit as st
from google import genai
from google.oauth2 import service_account


# ============================================================
# Page + CSS "pro"
# ============================================================
st.set_page_config(
    page_title="Puntos de mantenimiento Tractor",
    page_icon="🧰",
    layout="centered",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
.block-container {max-width: 1100px; padding-top: 2rem; padding-bottom: 2.5rem;}
/* Cards */
.card {
  background: rgba(255,255,255,0.05);
  border: 1px solid rgba(255,255,255,0.10);
  border-radius: 18px;
  padding: 16px 16px 10px 16px;
}
.card h3 {margin-top: 0.1rem;}
.small-muted {opacity: 0.75; font-size: 0.95rem;}
.hr {height:1px; background: rgba(255,255,255,0.10); margin: 14px 0 14px 0;}
/* Pills */
.pill {
  display:inline-block; padding: 3px 10px; border-radius: 999px;
  border: 1px solid rgba(255,255,255,0.12);
  background: rgba(255,255,255,0.06);
  font-size: 0.85rem; opacity: 0.9;
}
</style>
""",
    unsafe_allow_html=True,
)

st.title("🧰 Puntos de mantenimiento (por horas)")
st.markdown(
    '<div class="small-muted">Introduce <b>marca</b>, <b>modelo</b> y <b>horas</b>. La app genera un checklist por sistemas y un plan por intervalos.</div>',
    unsafe_allow_html=True,
)


# ============================================================
# Vertex / Gemini (solo Streamlit con st.secrets)
# ============================================================
DEFAULT_LOCATION = "us-central1"
DEFAULT_MODEL = "gemini-2.0-flash"

SYSTEMS = [
    "Motor y admisión",
    "Refrigeración",
    "Combustible",
    "Transmisión",
    "Hidráulico",
    "Frenos",
    "Dirección",
    "Eje delantero",
    "PTO/TDF",
    "Electricidad",
    "Cabina",
    "Engrase general",
    "Neumáticos",
]

PRESETS = {
    "Equilibrado (taller)": {
        "top_p": 0.9,
        "top_k": 40,
        "max_output_tokens": 2048,
    },
    "Más creativo (ideas + checklist)": {
        "top_p": 0.95,
        "top_k": 64,
        "max_output_tokens": 3072,
    },
    "Más conservador (menos inventar)": {
        "top_p": 0.8,
        "top_k": 20,
        "max_output_tokens": 2048,
    },
}


def conectar_vertex_desde_streamlit(location: str = DEFAULT_LOCATION) -> genai.Client:
    """
    Conexión a Vertex AI usando service account guardada en st.secrets["google"].
    Repara private_key si viene con \\n.
    """
    if "google" not in st.secrets:
        raise ValueError("No existe st.secrets['google']. Añade el bloque [google] en secrets.")

    creds_dict = st.secrets["google"]

    pk = str(creds_dict.get("private_key", ""))
    clean_key = pk.strip().strip('"').strip("'").replace("\\n", "\n")

    auth_info = {
        "type": "service_account",
        "project_id": creds_dict.get("project_id"),
        "private_key": clean_key,
        "client_email": creds_dict.get("client_email"),
        "token_uri": "https://oauth2.googleapis.com/token",
    }

    missing = [k for k in ("project_id", "private_key", "client_email") if not auth_info.get(k)]
    if missing:
        raise ValueError(f"Faltan campos en st.secrets['google']: {', '.join(missing)}")

    google_creds = service_account.Credentials.from_service_account_info(auth_info)
    scoped_creds = google_creds.with_scopes(["https://www.googleapis.com/auth/cloud-platform"])

    return genai.Client(
        vertexai=True,
        project=auth_info["project_id"],
        location=location,
        credentials=scoped_creds,
    )


@st.cache_resource(show_spinner=False)
def get_client() -> genai.Client:
    return conectar_vertex_desde_streamlit()


def build_prompt(marca: str, modelo: str, horas: int, objetivo: str) -> str:
    """
    Prompt: JSON estricto + defensivo (si no sabe, que lo diga).
    Incluimos 'fuentes' como lista de URLs o identificadores (si las conoce).
    """
    systems_txt = ", ".join([f'"{s}"' for s in SYSTEMS])

    return f"""
Eres un jefe de taller especialista en tractores agrícolas.
Tu objetivo: {objetivo}

Datos:
- Marca: {marca}
- Modelo: {modelo}
- Horas actuales: {horas}

Devuelve SOLO JSON válido (sin texto adicional, sin markdown), con esta estructura exacta:

{{
  "resumen": {{
    "marca": "{marca}",
    "modelo": "{modelo}",
    "horas": {horas},
    "intervalo_mas_cercano_h": 0,
    "razon_intervalo": "...",
    "confianza": "Alta | Media | Baja"
  }},
  "puntos_mantenimiento": [
    {{
      "sistema": "Motor y admisión",
      "items": [
        {{
          "tarea": "...",
          "tipo": "Sustitución | Inspección | Limpieza | Ajuste | Engrase",
          "prioridad": "Alta | Media | Baja",
          "frecuencia_h": 0,
          "tiempo_estimado_min": 0,
          "materiales": ["..."],
          "notas": "..."
        }}
      ]
    }}
  ],
  "ref_partes": [
    {{
      "pieza": "Filtro aceite motor",
      "referencia": "...",
      "motivo": "...",
      "confianza": "Alta | Media | Baja"
    }}
  ],
  "fuentes": [
    {{
      "titulo": "...",
      "url": "...",
      "nota": "si no tienes fuente real, deja url vacío y explica"
    }}
  ],
  "consumibles_recomendados": [
    {{
      "nombre": "Aceite motor 15W-40",
      "cantidad_aprox": "..."
    }}
  ],
  "chequeos_criticos": [
    {{
      "alerta": "...",
      "que_mirar": ["..."],
      "accion": "..."
    }}
  ],
  "suposiciones": ["..."]
}}

Reglas:
- Sistemas permitidos (usa EXACTAMENTE uno de estos): [{systems_txt}]
- Si no sabes intervalos exactos del modelo, usa intervalos típicos (250/500/1000/1500/2000h) y explica en "suposiciones".
- Evita números ultra específicos si no estás seguro; usa "aprox" y deja claro en notas/suposiciones.
- "ref_partes": si no puedes asegurar referencias, pon "confianza":"Baja" y explica el motivo; NO inventes con confianza alta.
- "fuentes": si no tienes URLs reales, deja "url":"" y explica en "nota" que no hay grounding.
""".strip()


def _strip_code_fences(text: str) -> str:
    t = (text or "").strip()
    if t.startswith("```"):
        t = t.strip("`").strip()
        if t.lower().startswith("json"):
            t = t[4:].strip()
    return t


def call_ai(
    marca: str,
    modelo: str,
    horas: int,
    model_name: str,
    temperature: float,
    seed: int,
    preset: dict,
    objetivo: str,
) -> dict:
    client = get_client()
    prompt = build_prompt(marca, modelo, horas, objetivo)

    # Config generación (según SDK puede variar; dejamos fallback)
    cfg = {
        "response_mime_type": "application/json",
        "temperature": float(temperature),
        "top_p": float(preset.get("top_p", 0.9)),
        "top_k": int(preset.get("top_k", 40)),
        "max_output_tokens": int(preset.get("max_output_tokens", 2048)),
        # seed: algunos SDK lo ignoran; lo dejamos por si está soportado
        "seed": int(seed),
    }

    try:
        resp = client.models.generate_content(
            model=model_name,
            contents=prompt,
            config=cfg,
        )
        text = _strip_code_fences(resp.text)
    except Exception:
        # Fallback conservador
        resp = client.models.generate_content(model=model_name, contents=prompt)
        text = _strip_code_fences(resp.text)

    try:
        return json.loads(text)
    except Exception as e:
        raise ValueError(
            f"No se pudo parsear JSON devuelto por IA. Error: {e}\n\nRespuesta IA:\n{text}"
        )


# ============================================================
# Session state
# ============================================================
if "history" not in st.session_state:
    st.session_state.history = []  # lista de dicts: {ts, inputs, data}
if "last_data" not in st.session_state:
    st.session_state.last_data = None


# ============================================================
# Sidebar: settings
# ============================================================
with st.sidebar:
    st.markdown("### ⚙️ Ajustes")

    preset_name = st.selectbox("Preset", list(PRESETS.keys()), index=0)
    preset = PRESETS[preset_name]

    st.markdown('<div class="hr"></div>', unsafe_allow_html=True)

    model_name = st.text_input("Modelo IA (Vertex)", value=DEFAULT_MODEL)
    location = st.text_input("Región Vertex", value=DEFAULT_LOCATION)

    st.markdown('<div class="hr"></div>', unsafe_allow_html=True)

    temperature = st.slider("Temperatura", 0.0, 1.5, 1.0,

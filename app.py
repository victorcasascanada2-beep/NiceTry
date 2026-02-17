# app.py
# Streamlit "estable" (sin sliders/presets), JSON robusto con autorreparación

import json
from datetime import datetime

import streamlit as st
from google import genai
from google.oauth2 import service_account


# ============================================================
# CONFIG FIJA (NO TOCAR)
# ============================================================
LOCATION = "us-central1"
MODEL_NAME = "gemini-2.0-flash"

GEN_CONFIG = {
    "response_mime_type": "application/json",
    "temperature": 0.6,          # estable (evita “delirios” y JSON rotos)
    "top_p": 0.85,
    "top_k": 40,
    "max_output_tokens": 4096,   # evita truncados
    "seed": 20240317,            # seed fija para repetibilidad
}

REPAIR_CONFIG = {
    "response_mime_type": "application/json",
    "temperature": 0.0,          # reparación determinista
    "max_output_tokens": 2048,
}

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


# ============================================================
# UI (look limpio + algo de CSS)
# ============================================================
st.set_page_config(page_title="Puntos de mantenimiento Tractor", page_icon="🧰", layout="centered")

st.markdown(
    """
<style>
.block-container {max-width: 1100px; padding-top: 1.6rem; padding-bottom: 2.2rem;}
#MainMenu, footer, header {visibility: hidden;}
.card {
  background: rgba(255,255,255,.04);
  border: 1px solid rgba(255,255,255,.10);
  border-radius: 18px;
  padding: 16px 16px 12px 16px;
}
.hero {
  background: radial-gradient(1100px circle at 20% 0%, rgba(124,58,237,.25), transparent 55%),
              radial-gradient(900px circle at 95% 10%, rgba(56,189,248,.18), transparent 45%),
              linear-gradient(180deg, rgba(255,255,255,.06), rgba(255,255,255,0));
  border: 1px solid rgba(255,255,255,.10);
  border-radius: 22px;
  padding: 18px 18px 14px 18px;
  margin-bottom: 16px;
}
.hero h1 {margin: 0 0 .2rem 0; line-height: 1.05;}
.hero p {margin: 0; opacity: .78;}
.pill {
  display:inline-block; padding: 4px 10px; border-radius: 999px;
  border: 1px solid rgba(255,255,255,.14);
  background: rgba(255,255,255,.06);
  font-size: .85rem; opacity:.92;
}
.stButton > button {
  border-radius: 14px !important;
  border: 1px solid rgba(255,255,255,.16) !important;
  padding: 0.7rem 1rem !important;
  font-weight: 700 !important;
  background: linear-gradient(135deg, rgba(124,58,237,.95), rgba(56,189,248,.65)) !important;
}
.stButton > button:hover {filter: brightness(1.06); transform: translateY(-1px);}
div[data-testid="stForm"] {
  border: 1px solid rgba(255,255,255,.10);
  border-radius: 18px;
  padding: 14px;
  background: rgba(255,255,255,.03);
}
</style>
""",
    unsafe_allow_html=True,
)

st.markdown(
    """
<div class="hero">
  <h1>🧰 Puntos de mantenimiento (por horas)</h1>
  <p>Introduce marca, modelo y horas. Checklist por sistemas + consumibles + críticos. Salida JSON descargable.</p>
</div>
""",
    unsafe_allow_html=True,
)

st.markdown(
    f"<span class='pill'>Modelo: {MODEL_NAME}</span> "
    f"<span class='pill'>Temp: {GEN_CONFIG['temperature']}</span> "
    f"<span class='pill'>Seed: {GEN_CONFIG['seed']}</span> "
    f"<span class='pill'>Tokens: {GEN_CONFIG['max_output_tokens']}</span>",
    unsafe_allow_html=True,
)


# ============================================================
# Vertex / Gemini (Streamlit secrets)
# ============================================================
def conectar_vertex_desde_streamlit(location: str = LOCATION) -> genai.Client:
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


# ============================================================
# Prompt (acotado para evitar truncados)
# ============================================================
def build_prompt(marca: str, modelo: str, horas: int) -> str:
    systems_txt = ", ".join([f'"{s}"' for s in SYSTEMS])

    return f"""
Eres un jefe de taller especialista en tractores agrícolas.
Devuelve SOLO JSON válido (sin texto adicional, sin markdown).

Datos:
- Marca: {marca}
- Modelo: {modelo}
- Horas actuales: {horas}

Estructura exacta:

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
      "url": "",
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
- Si no sabes intervalos exactos del modelo, usa intervalos típicos (250/500/1000/1500/2000h) y explícalo en "suposiciones".
- Máximo 4 tareas por sistema (prioriza las más relevantes).
- Evita cifras ultra específicas si no estás seguro; usa "aprox" y aclara en notas.
- NO inventes referencias con confianza alta: si dudas, pon confianza "Baja" y explica motivo.
- Si la respuesta empieza a ser larga, REDUCE contenido antes de romper el JSON. Cierra siempre llaves y listas.
""".strip()


# ============================================================
# JSON defensivo + reparación automática
# ============================================================
def _strip_code_fences(text: str) -> str:
    t = (text or "").strip()
    if t.startswith("```"):
        t = t.strip("`").strip()
        if t.lower().startswith("json"):
            t = t[4:].strip()
    return t


def _extract_json_object(text: str) -> str:
    """Coge desde el primer '{' hasta el último '}'."""
    t = (text or "").strip()
    a = t.find("{")
    b = t.rfind("}")
    if a != -1 and b != -1 and b > a:
        return t[a : b + 1]
    return t


def _repair_json_with_model(client: genai.Client, bad_text: str) -> str:
    """Pide al modelo que cierre/reponga el JSON sin cambiar estructura."""
    repair_prompt = f"""
El siguiente JSON está inválido o truncado. Arréglalo.

Reglas:
- Devuelve SOLO JSON válido (sin markdown, sin texto).
- Mantén la misma estructura y campos.
- Si falta el final, CIERRA correctamente listas y objetos.
- NO añadas explicaciones fuera del JSON.

JSON a reparar:
{bad_text}
""".strip()

    # intentamos forzar JSON
    try:
        resp = client.models.generate_content(
            model=MODEL_NAME,
            contents=repair_prompt,
            config=REPAIR_CONFIG,
        )
        return _strip_code_fences(resp.text)
    except Exception:
        resp = client.models.generate_content(model=MODEL_NAME, contents=repair_prompt)
        return _strip_code_fences(resp.text)


def call_ai(marca: str, modelo: str, horas: int) -> dict:
    client = get_client()
    prompt = build_prompt(marca, modelo, horas)

    # 1) llamada principal
    try:
        resp = client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt,
            config=GEN_CONFIG,
        )
        text = _strip_code_fences(resp.text)
    except Exception:
        # fallback si el SDK ignora config
        resp = client.models.generate_content(model=MODEL_NAME, contents=prompt)
        text = _strip_code_fences(resp.text)

    # 2) parse normal
    text2 = _extract_json_object(text)
    try:
        return json.loads(text2)
    except Exception:
        # 3) reparación
        fixed = _repair_json_with_model(client, text2)
        fixed2 = _extract_json_object(fixed)
        return json.loads(fixed2)


# ============================================================
# Session state
# ============================================================
if "history" not in st.session_state:
    st.session_state.history = []
if "last_data" not in st.session_state:
    st.session_state.last_data = None


# ============================================================
# FORM (inputs)
# ============================================================
left, right = st.columns([1.15, 0.85], gap="large")

with left:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 🧾 Datos del tractor")

    with st.form("mantenimiento", clear_on_submit=False):
        c1, c2 = st.columns(2)
        with c1:
            marca = st.text_input("Marca", placeholder="John Deere, New Holland, Case IH…")
        with c2:
            modelo = st.text_input("Modelo", placeholder="6120M, T7.230, Puma 150…")

        horas = st.number_input("Horas actuales", min_value=0, value=250, step=10)

        submit = st.form_submit_button("🚀 Calcular mantenimiento", use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)

with right:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 📌 Estado")

    if st.session_state.last_data:
        resumen = st.session_state.last_data.get("resumen", {}) or {}
        st.metric("Confianza", resumen.get("confianza", "—"))
        st.metric("Intervalo cercano (h)", resumen.get("intervalo_mas_cercano_h", "—"))
    else:
        st.info("Rellena el formulario y pulsa **Calcular**.")

    if st.button("🧹 Borrar historial", use_container_width=True):
        st.session_state.history = []
        st.session_state.last_data = None
        st.success("Historial borrado.")

    st.markdown("</div>", unsafe_allow_html=True)


# ============================================================
# RUN
# ============================================================
if submit:
    if not marca.strip() or not modelo.strip():
        st.error("Faltan datos: **marca** y **modelo** son obligatorios.")
        st.stop()

    with st.status("Generando plan de mantenimiento…", expanded=True) as status:
        st.write("🧠 Llamando a Vertex…")
        try:
            data = call_ai(marca.strip(), modelo.strip(), int(horas))
        except Exception as e:
            status.update(label="Error", state="error")
            st.error(str(e))
            st.stop()

        status.update(label="Listo", state="complete")

    st.session_state.last_data = data
    st.session_state.history.insert(
        0,
        {
            "ts": datetime.now().isoformat(timespec="seconds"),
            "inputs": {"marca": marca.strip(), "modelo": modelo.strip(), "horas": int(horas)},
            "data": data,
        },
    )


# ============================================================
# OUTPUT (tabs)
# ============================================================
data = st.session_state.last_data
tabs = st.tabs(["✅ Checklist", "🧾 Resumen", "🧩 Partes & fuentes", "📦 Consumibles", "⚠️ Críticos", "🧠 Suposiciones", "🧬 JSON", "🕘 Historial"])

with tabs[0]:
    if not data:
        st.info("Sin resultados todavía.")
    else:
        pm = data.get("puntos_mantenimiento", []) or []
        if not pm:
            st.warning("No llegaron puntos de mantenimiento.")
        else:
            for bloque in pm:
                sistema = bloque.get("sistema", "Sistema")
                items = bloque.get("items", []) or []
                st.markdown(f"### {sistema}")
                for it in items:
                    tarea = it.get("tarea", "Tarea")
                    prioridad = it.get("prioridad", "")
                    tipo = it.get("tipo", "")
                    freq = it.get("frecuencia_h", "")
                    tmin = it.get("tiempo_estimado_min", "")
                    materiales = it.get("materiales", []) or []
                    notas = it.get("notas", "")

                    cols = st.columns([0.06, 0.94])
                    with cols[0]:
                        st.checkbox("", value=False, key=f"chk_{sistema}_{tarea}_{freq}_{tmin}")
                    with cols[1]:
                        st.markdown(
                            f"**{tarea}**  \n"
                            f"<span class='pill'>Tipo: {tipo}</span> "
                            f"<span class='pill'>Prioridad: {prioridad}</span> "
                            f"<span class='pill'>Frecuencia(h): {freq}</span> "
                            f"<span class='pill'>Tiempo(min): {tmin}</span>",
                            unsafe_allow_html=True,
                        )
                        if materiales:
                            st.caption("Materiales: " + ", ".join([str(x) for x in materiales]))
                        if notas:
                            st.caption(notas)

with tabs[1]:
    if not data:
        st.info("Sin resultados todavía.")
    else:
        st.subheader("Resumen")
        st.json(data.get("resumen", {}) or {})

with tabs[2]:
    if not data:
        st.info("Sin resultados todavía.")
    else:
        st.subheader("Ref. de partes")
        st.json(data.get("ref_partes", []) or [])
        st.subheader("Fuentes")
        st.json(data.get("fuentes", []) or [])
        st.warning("Si 'fuentes.url' viene vacío, NO hay grounding real: referencias aproximadas.")

with tabs[3]:
    if not data:
        st.info("Sin resultados todavía.")
    else:
        st.subheader("Consumibles recomendados")
        st.json(data.get("consumibles_recomendados", []) or [])

with tabs[4]:
    if not data:
        st.info("Sin resultados todavía.")
    else:
        st.subheader("Chequeos críticos")
        st.json(data.get("chequeos_criticos", []) or [])

with tabs[5]:
    if not data:
        st.info("Sin resultados todavía.")
    else:
        st.subheader("Suposiciones")
        st.json(data.get("suposiciones", []) or [])

with tabs[6]:
    if not data:
        st.info("Sin resultados todavía.")
    else:
        st.subheader("Salida JSON completa")
        st.code(json.dumps(data, ensure_ascii=False, indent=2), language="json")

        resumen = data.get("resumen", {}) or {}
        fn = f"mantenimiento_{resumen.get('marca','marca')}_{resumen.get('modelo','modelo')}_{resumen.get('horas','horas')}h.json"
        fn = fn.replace(" ", "_")

        st.download_button(
            "⬇️ Descargar JSON",
            data=json.dumps(data, ensure_ascii=False, indent=2),
            file_name=fn,
            mime="application/json",
            use_container_width=True,
        )

with tabs[7]:
    st.subheader("Historial")
    if not st.session_state.history:
        st.info("No hay historial todavía.")
    else:
        for i, item in enumerate(st.session_state.history[:20], start=1):
            inputs = item.get("inputs", {})
            ts = item.get("ts", "")
            with st.expander(f"{i}. {inputs.get('marca','?')} {inputs.get('modelo','?')} — {inputs.get('horas','?')}h · {ts}", expanded=False):
                st.json(inputs)
                st.code(json.dumps(item.get("data", {}), ensure_ascii=False, indent=2), language="json")

st.divider()
st.caption("Requiere Streamlit Secrets: bloque [google] con project_id, client_email y private_key.")

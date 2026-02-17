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

    temperature = st.slider("Temperatura", 0.0, 1.5, 1.0, 0.05)
    seed_mode = st.radio("Seed", ["Aleatoria (cada ejecución)", "Fija"], horizontal=True)

    if seed_mode == "Fija":
        seed = st.number_input("Seed fija", min_value=0, max_value=2_147_483_647, value=123456, step=1)
    else:
        seed = random.randint(0, 2_147_483_647)

    objetivo = st.text_area(
        "Objetivo (estilo de salida)",
        value="Checklist claro para taller, con prioridades y consumibles. Sé realista: si no sabes algo, dilo.",
        height=90,
    )

    st.markdown('<div class="hr"></div>', unsafe_allow_html=True)

    if st.button("🧹 Borrar historial", use_container_width=True):
        st.session_state.history = []
        st.session_state.last_data = None
        st.success("Historial borrado.")


# ============================================================
# Main layout: input card + output tabs
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

        c3, c4 = st.columns([0.55, 0.45])
        with c3:
            submit = st.form_submit_button("🚀 Calcular mantenimiento", use_container_width=True)
        with c4:
            st.markdown(
                f'<span class="pill">Preset: {preset_name}</span> '
                f'<span class="pill">Temp: {temperature}</span> '
                f'<span class="pill">Seed: {seed}</span>',
                unsafe_allow_html=True,
            )

    st.markdown('</div>', unsafe_allow_html=True)

with right:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 📌 Estado")
    st.caption("Validación, métricas y avisos")

    missing = []
    if "marca" not in st.session_state:
        pass

    if st.session_state.last_data:
        resumen = st.session_state.last_data.get("resumen", {}) or {}
        conf = resumen.get("confianza", "—")
        intervalo = resumen.get("intervalo_mas_cercano_h", "—")
        st.metric("Confianza", conf)
        st.metric("Intervalo cercano (h)", intervalo)
    else:
        st.info("Aún no hay resultados. Rellena el formulario y pulsa **Calcular**.")

    st.markdown('</div>', unsafe_allow_html=True)


# ============================================================
# Run
# ============================================================
if submit:
    if not marca.strip() or not modelo.strip():
        st.error("Faltan datos: **marca** y **modelo** son obligatorios.")
        st.stop()

    # Re-conectar si el usuario cambia región (cache_resource usa DEFAULT_LOCATION).
    # Para no complicar: avisamos si cambió.
    if location.strip() != DEFAULT_LOCATION:
        st.warning(
            f"Has cambiado la región a **{location}**, pero el cliente cacheado usa **{DEFAULT_LOCATION}**. "
            "Si necesitas otra región, cambia DEFAULT_LOCATION en el código y redeploy."
        )

    with st.status("Generando plan de mantenimiento…", expanded=True) as status:
        st.write("📨 Preparando prompt…")
        time.sleep(0.05)
        st.write("🧠 Llamando a Vertex…")
        try:
            data = call_ai(
                marca.strip(),
                modelo.strip(),
                int(horas),
                model_name.strip(),
                temperature=float(temperature),
                seed=int(seed),
                preset=preset,
                objetivo=objetivo.strip(),
            )
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
            "inputs": {
                "marca": marca.strip(),
                "modelo": modelo.strip(),
                "horas": int(horas),
                "model": model_name.strip(),
                "temperature": float(temperature),
                "seed": int(seed),
                "preset": preset_name,
            },
            "data": data,
        },
    )


# ============================================================
# Output
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
        st.subheader("Fuentes (si existen)")
        st.json(data.get("fuentes", []) or [])
        st.warning("Si 'fuentes.url' viene vacío, NO hay grounding real: las referencias pueden ser aproximadas.")

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

        st.download_button(
            "⬇️ Descargar JSON",
            data=json.dumps(data, ensure_ascii=False, indent=2),
            file_name=f"mantenimiento_{(data.get('resumen', {}).get('marca','') or 'marca')}_{(data.get('resumen', {}).get('modelo','') or 'modelo')}_{(data.get('resumen', {}).get('horas','') or 'horas')}h.json".replace(" ", "_"),
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
            with st.expander(f"{i}. {inputs.get('marca','?')} {inputs.get('modelo','?')} — {inputs.get('horas','?')}h  ·  {ts}", expanded=False):
                st.json(inputs)
                st.code(json.dumps(item.get("data", {}), ensure_ascii=False, indent=2), language="json")

st.divider()
st.caption("Requiere Streamlit Secrets: bloque [google] con project_id, client_email y private_key.")

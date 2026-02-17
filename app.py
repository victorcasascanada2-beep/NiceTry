
import json
import streamlit as st

from google import genai
from google.oauth2 import service_account

# ============================================================
# UI
# ============================================================
st.set_page_config(page_title="Puntos de mantenimiento Tractor", page_icon="🧰", layout="centered")
st.title("🧰 Puntos de mantenimiento (por horas)")
st.caption("Introduce marca, modelo y horas. La app genera un checklist y un plan por intervalos.")

# ============================================================
# Vertex / Gemini (solo Streamlit con st.secrets)
# ============================================================
DEFAULT_LOCATION = "us-central1"  # cámbialo si tu Vertex está en otra región
DEFAULT_MODEL = "gemini-2.0-flash"  # cámbialo si usas otro (p.ej. gemini-1.5-pro)

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


def build_prompt(marca: str, modelo: str, horas: int) -> str:
    return f"""
Eres un jefe de taller especialista en tractores agrícolas.

Quiero que calcules el mantenimiento que corresponde con:
- Marca: {marca}
- Modelo: {modelo}
- Horas actuales: {horas}
- Investiga en google y dime la referencia de la parte a montar.
Entregable (en ESPAÑOL) y SOLO en JSON válido (sin texto adicional), con esta estructura exacta:

{{
  "resumen": {{
    "marca": "...",
    "modelo": "...",
    "horas": 0,
    "intervalo_mas_cercano_h": 0,
    "razon_intervalo": "...",
    "Ref de partes": "..."
  }},
  "puntos_mantenimiento": [
    {{
      "sistema": "Motor y admisión",
      "items": [
        {{
          "tarea": "Cambiar aceite motor",
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
  "consumibles_recomendados": [
    {{
      "nombre": "Aceite motor 15W-40",
      "cantidad_aprox": "..."
    }}
  ],
  "chequeos_criticos": [
    {{
      "alerta": "Sobrecalentamiento",
      "que_mirar": ["..."],
      "accion": "..."
    }}
  ],
  "suposiciones": ["Si falta info, indica aquí lo supuesto (ej. intervalo típico 500h, etc.)"]
}}

Reglas:
- Si no sabes el intervalo exacto del modelo, usa intervalos típicos (250/500/1000/1500/2000h) y explica en "suposiciones".
- Agrupa en sistemas: Motor/Admisión, Refrigeración, Combustible, Transmisión, Hidráulico, Frenos, Dirección, Eje delantero, PTO/TDF, Electricidad, Cabina, Engrase general, Neumáticos.
- Incluye tareas realistas: filtros (aire/combustible/hidráulico), correas, refrigerante, engrase, niveles, holguras, fugas, diagnosis básica.
- Evita cifras ultra específicas si no estás seguro; usa "aprox" y deja claro en notas/suposiciones.
"""


@st.cache_resource(show_spinner=False)
def get_client() -> genai.Client:
    return conectar_vertex_desde_streamlit()


def call_ai(marca: str, modelo: str, horas: int, model_name: str) -> dict:
    client = get_client()
    prompt = build_prompt(marca, modelo, horas)

    # Salida en JSON estricta (si tu SDK/versión no soporta response_mime_type, hacemos fallback).
    try:
        resp = client.models.generate_content(
            model=model_name,
            contents=prompt,
            config={"response_mime_type": "application/json"},
        )
        text = (resp.text or "").strip()
    except Exception:
        resp = client.models.generate_content(model=model_name, contents=prompt)
        text = (resp.text or "").strip()

    # Limpieza defensiva (por si el modelo mete ```json ... ```).
    if text.startswith("```"):
        text = text.strip("`")
        # a veces queda "json\n{...}"
        if text.lower().startswith("json"):
            text = text[4:].strip()

    try:
        return json.loads(text)
    except Exception as e:
        raise ValueError(f"No se pudo parsear JSON devuelto por IA. Error: {e}\n\nRespuesta IA:\n{text}")


# ============================================================
# FORM
# ============================================================
with st.form("mantenimiento"):
    c1, c2 = st.columns(2)
    with c1:
        marca = st.text_input("Marca", placeholder="John Deere, New Holland, Case IH…")
    with c2:
        modelo = st.text_input("Modelo", placeholder="6120M, T7.230, Puma 150…")

    horas = st.number_input("Horas actuales", min_value=0, value=250, step=10)

    # Opcional: permitir cambiar modelo Gemini sin tocar código
    model_name = st.text_input("Modelo IA (Vertex)", value=DEFAULT_MODEL)

    submit = st.form_submit_button("Calcular puntos de mantenimiento")

# ============================================================
# RUN
# ============================================================
if submit:
    if not marca.strip() or not modelo.strip():
        st.error("Faltan datos: marca y modelo son obligatorios.")
        st.stop()

    with st.spinner("Calculando…"):
        try:
            data = call_ai(marca.strip(), modelo.strip(), int(horas), model_name.strip())
        except Exception as e:
            st.error(str(e))
            st.stop()

    # Render
    st.subheader("Resumen")
    st.json(data.get("resumen", {}))

    st.subheader("Puntos de mantenimiento")
    for bloque in data.get("puntos_mantenimiento", []):
        sistema = bloque.get("sistema", "Sistema")
        st.markdown(f"### {sistema}")
        items = bloque.get("items", [])
        for it in items:
            tarea = it.get("tarea", "Tarea")
            prioridad = it.get("prioridad", "")
            tipo = it.get("tipo", "")
            freq = it.get("frecuencia_h", "")
            tmin = it.get("tiempo_estimado_min", "")
            materiales = it.get("materiales", [])
            notas = it.get("notas", "")

            st.markdown(f"- **{tarea}**  \n"
                        f"  - Tipo: `{tipo}` | Prioridad: `{prioridad}` | Frecuencia(h): `{freq}` | Tiempo(min): `{tmin}`")
            if materiales:
                st.markdown(f"  - Materiales: {', '.join([str(x) for x in materiales])}")
            if notas:
                st.markdown(f"  - Notas: {notas}")

    st.subheader("Consumibles recomendados")
    st.json(data.get("consumibles_recomendados", []))

    st.subheader("Chequeos críticos")
    st.json(data.get("chequeos_criticos", []))

    st.subheader("Suposiciones")
    st.json(data.get("suposiciones", []))

    # Export rápido
    st.download_button(
        "Descargar JSON",
        data=json.dumps(data, ensure_ascii=False, indent=2),
        file_name=f"mantenimiento_{marca}_{modelo}_{horas}h.json".replace(" ", "_"),
        mime="application/json",
    )

st.divider()
st.caption("Nota: asegúrate de tener en Streamlit Secrets un bloque [google] con project_id, client_email y private_key.")

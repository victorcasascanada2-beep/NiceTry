
import os
import json
from nicegui import ui

from google import genai
from google.oauth2 import service_account

DEFAULT_LOCATION = os.getenv("VERTEX_LOCATION", "us-central1")
DEFAULT_MODEL = os.getenv("VERTEX_MODEL", "gemini-2.0-flash")


def _get_google_creds_from_env() -> service_account.Credentials:
    """
    Lee credenciales desde variables de entorno (para NiceGUI / Cloud Run).
    Recomendado: definir GOOGLE_SA_JSON como JSON completo del service account.
    Alternativa: project_id/client_email/private_key por separado.
    """
    sa_json = os.getenv("GOOGLE_SA_JSON", "").strip()

    if sa_json:
        info = json.loads(sa_json)
        pk = str(info.get("private_key", ""))
        info["private_key"] = pk.replace("\\n", "\n")
        creds = service_account.Credentials.from_service_account_info(info)
        return creds.with_scopes(["https://www.googleapis.com/auth/cloud-platform"])

    project_id = os.getenv("GOOGLE_PROJECT_ID", "").strip()
    client_email = os.getenv("GOOGLE_CLIENT_EMAIL", "").strip()
    private_key = os.getenv("GOOGLE_PRIVATE_KEY", "").strip().replace("\\n", "\n")

    missing = [k for k, v in {
        "GOOGLE_PROJECT_ID": project_id,
        "GOOGLE_CLIENT_EMAIL": client_email,
        "GOOGLE_PRIVATE_KEY": private_key,
    }.items() if not v]

    if missing:
        raise ValueError(f"Faltan credenciales. Define GOOGLE_SA_JSON o estas vars: {', '.join(missing)}")

    info = {
        "type": "service_account",
        "project_id": project_id,
        "private_key": private_key,
        "client_email": client_email,
        "token_uri": "https://oauth2.googleapis.com/token",
    }
    creds = service_account.Credentials.from_service_account_info(info)
    return creds.with_scopes(["https://www.googleapis.com/auth/cloud-platform"])


def get_client() -> genai.Client:
    creds = _get_google_creds_from_env()
    # project en Client = project_id de la SA
    project_id = creds.service_account_email.split("@")[-1]  # fallback feo si no hay project; lo evitamos abajo
    # mejor: si usas GOOGLE_SA_JSON / GOOGLE_PROJECT_ID, lo tienes:
    project_id = os.getenv("GOOGLE_PROJECT_ID", "") or json.loads(os.getenv("GOOGLE_SA_JSON", "{}") or "{}").get("project_id")
    if not project_id:
        raise ValueError("No se pudo determinar project_id. Define GOOGLE_PROJECT_ID o GOOGLE_SA_JSON con project_id.")

    return genai.Client(vertexai=True, project=project_id, location=DEFAULT_LOCATION, credentials=creds)


def build_prompt(marca: str, modelo: str, horas: int) -> str:
    # Es tu prompt actual simplificado; puedes copiarlo completo desde app.py si quieres.
    return f"""
Eres un jefe de taller especialista en tractores agrícolas.

Quiero que calcules el mantenimiento que corresponde con:
- Marca: {marca}
- Modelo: {modelo}
- Horas actuales: {horas}

Entregable (en ESPAÑOL) y SOLO en JSON válido (sin texto adicional), con esta estructura:

{{
  "resumen": {{
    "marca": "...",
    "modelo": "...",
    "horas": 0,
    "intervalo_mas_cercano_h": 0,
    "razon_intervalo": "..."
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
  ]
}}
"""


def call_ai(marca: str, modelo: str, horas: int, model_name: str) -> dict:
    client = get_client()
    prompt = build_prompt(marca, modelo, horas)

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

    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:].strip()

    return json.loads(text)


# ---------------- UI (NiceGUI) ----------------
ui.label("🧰 Puntos de mantenimiento (NiceGUI)").classes("text-h5")

with ui.row().classes("w-full"):
    marca_in = ui.input(label="Marca", placeholder="John Deere, New Holland, Case IH…").classes("w-1/3")
    modelo_in = ui.input(label="Modelo", placeholder="6120M, T7.230, Puma 150…").classes("w-1/3")
    horas_in = ui.number(label="Horas actuales", value=250, min=0, precision=0).classes("w-1/3")

model_in = ui.input(label="Modelo IA (Vertex)", value=DEFAULT_MODEL)

out = ui.textarea(label="Salida JSON", placeholder="Aquí saldrá el JSON…").props("readonly").classes("w-full")


def on_click():
    marca = (marca_in.value or "").strip()
    modelo = (modelo_in.value or "").strip()
    horas = int(horas_in.value or 0)
    model_name = (model_in.value or DEFAULT_MODEL).strip()

    if not marca or not modelo:
        out.value = "ERROR: marca y modelo son obligatorios"
        return

    out.value = "Calculando…"
    try:
        data = call_ai(marca, modelo, horas, model_name)
        out.value = json.dumps(data, ensure_ascii=False, indent=2)
    except Exception as e:
        out.value = f"ERROR:\n{e}"


ui.button("Calcular puntos de mantenimiento", on_click=on_click)

port = int(os.getenv("PORT", "8080"))
ui.run(host="0.0.0.0", port=port)

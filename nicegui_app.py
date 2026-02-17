import os
from nicegui import ui

def doblar():
    try:
        salida.value = str(float(entrada.value) * 2)
    except Exception:
        salida.value = "Valor inválido"

ui.label("Demo NiceGUI")
entrada = ui.input(label="Valor", value="0")
ui.button("Doblar", on_click=doblar)
salida = ui.input(label="Resultado").props("readonly")

port = int(os.getenv("PORT", "8080"))
ui.run(host="0.0.0.0", port=port)

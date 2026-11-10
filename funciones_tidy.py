"""Herramientas para cargar resultados de Tidy3D y animar un monitor temporal.

Dependencias: tidy3d, numpy, matplotlib y pillow.
La API key se configura por separado; este módulo no guarda credenciales.
Importar este archivo no descarga datos ni ejecuta simulaciones.
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import tidy3d.web as web

__all__ = ["importar_datos", "generar_gif"]


def importar_datos(identificador):
    """Carga una tarea existente por Task ID y muestra sus monitores.

    Requiere una API key configurada con acceso a la tarea.
    Devuelve sim_data; no vuelve a ejecutar la simulación.
    """
    sim_data = web.load(identificador)
    print("Monitores disponibles:")
    for resultado in sim_data.data:
        print(f"  - {resultado.monitor.name}")
    return sim_data


def generar_gif(sim_data, monitor="fieldtimemonitor_0",
                archivo="pulso_completo.gif", fps=30, *,
                componente="Ex", eje_corte="x", posicion=0,
                cmap="RdBu_r", dpi=120):
    """Guarda todos los tiempos del campo real en un GIF con escala fija.

    Por defecto anima Ex en el plano yz más cercano a x=0.
    componente: Ex, Ey, Ez, Hx, Hy o Hz (debe estar registrado).
    eje_corte: x, y o z; posicion está en micrómetros.
    archivo: ruta del GIF. fps: fotogramas por segundo de reproducción.
    Devuelve la ruta del archivo generado. No muestra ni descarga el GIF.
    """
    if eje_corte not in ("x", "y", "z"):
        raise ValueError("eje_corte debe ser 'x', 'y' o 'z'.")
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("fps debe ser positivo y finito.")
    if componente not in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz"):
        raise ValueError("Componente no válida: use Ex, Ey, Ez, Hx, Hy o Hz.")
    campo = getattr(sim_data[monitor], componente, None)
    if campo is None:
        raise ValueError(f"El monitor {monitor!r} no contiene {componente}.")
    if "t" not in campo.dims:
        raise ValueError("Seleccione un monitor temporal (FieldTimeMonitor).")

    horizontal, vertical = [eje for eje in ("x", "y", "z")
                            if eje != eje_corte]
    campo = (campo.sel({eje_corte: posicion}, method="nearest")
             .real.transpose("t", vertical, horizontal))
    datos = campo.values
    tiempos = campo.t.values
    if len(tiempos) == 0 or not np.all(np.isfinite(datos)):
        raise ValueError("El campo está vacío o contiene valores no finitos.")
    if campo.sizes[horizontal] < 2 or campo.sizes[vertical] < 2:
        raise ValueError("Se necesita un monitor con extensión en los dos ejes del plano.")
    limite = float(np.max(np.abs(datos))) or 1.0
    corte_real = float(campo.coords[eje_corte])
    ruta = Path(archivo)
    if ruta.suffix.lower() != ".gif":
        raise ValueError("El archivo de salida debe terminar en .gif.")

    fig, ax = plt.subplots(figsize=(5, 7))
    try:
        imagen = ax.pcolormesh(
            campo[horizontal].values, campo[vertical].values, datos[0],
            shading="auto", cmap=cmap, vmin=-limite, vmax=limite)
        ax.set_xlabel(f"{horizontal} (µm)")
        ax.set_ylabel(f"{vertical} (µm)")
        ax.set_aspect("equal")
        fig.colorbar(imagen, ax=ax, label=f"Re({componente})")
        titulo = ax.set_title("")
        fig.tight_layout()

        def actualizar(i):
            imagen.set_array(datos[i].ravel())
            titulo.set_text(
                f"{eje_corte} = {corte_real:g} µm | t = {tiempos[i]*1e15:.2f} fs")
            return imagen, titulo

        animacion = FuncAnimation(fig, actualizar, frames=len(tiempos),
                                  interval=1000/fps, blit=False)
        animacion.save(str(ruta), writer=PillowWriter(fps=fps), dpi=dpi)
    finally:
        plt.close(fig)
    print(f"Guardado: {ruta} ({len(tiempos)} fotogramas)")
    return str(ruta)

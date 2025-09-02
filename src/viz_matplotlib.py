import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

def show_slider(times, lat_grid, lon_grid, cube,
                lat_min, lon_min, lat_max, lon_max):
    """
    Interactive matplotlib viewer with a time slider.
    (Keep separate from CLI so headless envs don’t break.)
    """
    idx0 = 0
    vmin, vmax = 0.0, float(np.nanmax(cube)) if np.isfinite(cube).any() else 1.0
    fig, ax = plt.subplots(figsize=(9, 7))
    plt.subplots_adjust(bottom=0.18)
    im = ax.imshow(
        cube[idx0], origin="lower",
        extent=[lon_grid[0], lon_grid[-1], lat_grid[0], lat_grid[-1]],
        aspect="auto", vmin=vmin, vmax=vmax
    )
    fig.colorbar(im, ax=ax, label="Hourly precipitation (mm)")
    title = ax.set_title(f"Rainfall @ {times[idx0].strftime('%Y-%m-%d %H:%M')} (UTC+8)")
    ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")

    slider_ax = fig.add_axes([0.12, 0.07, 0.76, 0.03])
    s_time = Slider(ax=slider_ax, label="Time index", valmin=0, valmax=len(times)-1, valinit=idx0, valstep=1)

    def update(_):
        k = int(s_time.val)
        im.set_data(cube[k])
        im.set_clim(vmin, vmax)
        title.set_text(f"Rainfall @ {times[k].strftime('%Y-%m-%d %H:%M')} (UTC+8)")
        fig.canvas.draw_idle()

    s_time.on_changed(update)
    plt.show()

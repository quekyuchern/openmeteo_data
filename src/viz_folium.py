import folium
from folium.plugins import HeatMapWithTime
import branca.colormap as cm

def make_folium_map(times, lat_grid, lon_grid, cube,
                    lat_min, lon_min, lat_max, lon_max):
    """
    Build a Folium HeatMapWithTime map object.
    """
    from .transform import build_folium_frames
    frames, index = build_folium_frames(times, lat_grid, lon_grid, cube)

    center = [(lat_min + lat_max) / 2.0, (lon_min + lon_max) / 2.0]
    m = folium.Map(location=center, zoom_start=11, tiles="CartoDB positron")

    HeatMapWithTime(
        frames,
        index=index,
        radius=22,
        max_opacity=0.9,
        auto_play=False,
        use_local_extrema=False
    ).add_to(m)

    vmax_global = float(cube[~(cube != cube)].max()) if cube.size else 1.0  # safe fallback
    legend = cm.linear.Blues_09.scale(0, vmax_global if vmax_global > 0 else 1.0)
    legend.caption = "Hourly precipitation (mm)"
    legend.add_to(m)

    m.fit_bounds([[lat_min, lon_min], [lat_max, lon_max]])
    return m

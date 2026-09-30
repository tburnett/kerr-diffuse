"""sky-plot display, with helpers for labels and hover tooltips."""
# %matplotlib widget
# from vscdark import set_style
# set_style(dark=True, display_css=True)

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from astropy.coordinates import SkyCoord
from utilities.skymaps import AITfigure, ZEAfigure


def _catalog_source_summary(row, *, in_model=False):
    """Build a short tooltip summary for a catalog source row."""
    lines = []

    name = row.name if hasattr(row, 'name') else None
    if isinstance(name, str) and name:
        lines.append(name)

    class_name = row.get('class1') if hasattr(row, 'get') else None
    if pd.notna(class_name):
        lines.append(f'class: {class_name}')

    assoc_name = None
    if hasattr(row, 'get'):
        assoc_name = row.get('assoc1_name', row.get('assoc1'))
    if pd.notna(assoc_name):
        lines.append(f'assoc: {assoc_name}')

    significance = row.get('significance') if hasattr(row, 'get') else None
    if pd.notna(significance):
        lines.append(f'significance: {float(significance):.1f}')

    ts = row.get('ts') if hasattr(row, 'get') else None
    if pd.notna(ts):
        lines.append(f'TS: {float(ts):.1f}')

    sep = row.get('sep') if hasattr(row, 'get') else None
    if pd.notna(sep):
        lines.append(f'sep: {float(sep):.2f} deg')

    specfunc = row.get('specfunc') if hasattr(row, 'get') else None
    if specfunc is not None:
        lines.append(f'model: {specfunc.__class__.__name__}')

    if in_model:
        lines.append('in source model')

    return '\n'.join(lines)


def _install_text_hover(ax, entries):
    """Attach hover annotations for text artists when the backend supports events."""
    if not entries:
        return None

    annotation = ax.annotate(
        '',
        xy=(0, 0),
        xytext=(10, 10),
        textcoords='offset points',
        ha='left',
        va='bottom',
        fontsize=8,
        color='white',
        bbox=dict(boxstyle='round', fc='black', ec='white', alpha=0.85),
        zorder=20,
    )
    annotation.set_visible(False)

    def on_move(event):
        if event.inaxes is not ax:
            if annotation.get_visible():
                annotation.set_visible(False)
            _set_colorbar_visible(ax, True)
            ax.figure.canvas.draw_idle()
            return

        for artist, summary in entries:
            contains, _ = artist.contains(event)
            if not contains:
                continue
            annotation.xy = artist.get_position()
            annotation.set_text(summary)
            annotation.set_visible(True)
            _set_colorbar_visible(ax, False)
            ax.figure.canvas.draw_idle()
            return

        if annotation.get_visible():
            annotation.set_visible(False)
        _set_colorbar_visible(ax, True)
        ax.figure.canvas.draw_idle()

    callback_id = ax.figure.canvas.mpl_connect('motion_notify_event', on_move)
    ax._hover_annotation = annotation
    ax._hover_entries = entries
    ax._hover_callback_id = callback_id
    return callback_id


def _format_hover_value(value, value_format='auto'):
    """Format a plotted value for a hover annotation."""
    if isinstance(value, (np.ndarray, list, tuple)):
        arr = np.asarray(value)
        if arr.size == 0:
            return str(value)
        if arr.size != 1:
            flat = arr.ravel()
            if np.allclose(flat, flat[0], rtol=1e-12, atol=1e-12):
                value = flat[0]
            else:
                value = np.nanmean(flat)
        else:
            value = arr.item()

    if value_format == 'auto':
        if isinstance(value, (np.floating, float)):
            if not np.isfinite(value):
                return str(value)
            if abs(value) >= 1e4 or (abs(value) > 0 and abs(value) < 1e-3):
                return f'{value:.2e}'
            text = f'{value:.3g}'
            if '.' in text:
                text = text.rstrip('0').rstrip('.')
            return text

        if isinstance(value, (np.integer, int)):
            return str(int(value))

        return str(value)

    try:
        return value_format.format(value)
    except (TypeError, ValueError):
        return str(value)


def _set_colorbar_visible(ax, visible):
    """Toggle the associated colorbar visibility for the figure."""

    # assume the colorbar axis is the last axis in the figure
    cbar_ax = ax.figure.axes[-1]
    if hasattr(cbar_ax, '_colorbar'):
        cbar_ax.set_visible(visible)
        ax.figure.canvas.draw_idle()


def _install_value_hover(ax, artist, value_format='auto'):
    """Attach a cursor-value hover annotation showing coordinates and image value."""
    if artist is None:
        return None

    annotation = ax.annotate(
        '',
        xy=(0, 0),
        xytext=(12, 12),
        textcoords='offset points',
        ha='left',
        va='bottom',
        fontsize=8,
        color='white',
        bbox=dict(boxstyle='round', fc='black', ec='white', alpha=0.85),
        zorder=25,
    )
    annotation.set_visible(False)

    def on_move(event):
        text_hover = getattr(ax, '_hover_annotation', None)
        if text_hover is not None and text_hover.get_visible():
            if annotation.get_visible():
                annotation.set_visible(False)
            _set_colorbar_visible(ax, True)
            ax.figure.canvas.draw_idle()
            return

        if event.inaxes is not ax or event.xdata is None or event.ydata is None:
            if annotation.get_visible():
                annotation.set_visible(False)
            _set_colorbar_visible(ax, True)
            ax.figure.canvas.draw_idle()
            return

        value = artist.get_cursor_data(event)
        if value is None:
            if annotation.get_visible():
                annotation.set_visible(False)
            _set_colorbar_visible(ax, True)
            ax.figure.canvas.draw_idle()
            return

        _set_colorbar_visible(ax, False)
        coord = ax.format_coord(event.xdata, event.ydata)
        annotation.xy = (event.xdata, event.ydata)
        annotation.set_text(f'{coord[:coord.find("(")]}: {_format_hover_value(value, value_format)}')
        annotation.set_visible(True)
        ax.figure.canvas.draw_idle()

    callback_id = ax.figure.canvas.mpl_connect('motion_notify_event', on_move)
    ax._value_hover_annotation = annotation
    ax._value_hover_artist = artist
    ax._value_hover_callback_id = callback_id
    return callback_id

# example usage:
# entries = [(text_artist, _catalog_source_summary(row)) for text_artist, row in zip(text_artists, catalog_rows)]
# _install_text_hover(ax, entries)          

# Make sure you have ipympl installed:
# pip install ipympl

# Activate the interactive matplotlib widget backend
# %matplotlib widget

# import numpy as np
# import matplotlib.pyplot as plt
# from ipywidgets import interact, FloatSlider

# # Create some data
# x = np.linspace(0, 2 * np.pi, 500)

# # Create the figure and axis
# fig, ax = plt.subplots()
# line, = ax.plot(x, np.sin(x), lw=2)
# ax.set_title("Interactive Sine Wave")
# ax.set_xlabel("x")
# ax.set_ylabel("y")

# # Update function for the slider
# def update(freq=1.0, amplitude=1.0):
#     """Update the sine wave based on slider values."""
#     line.set_ydata(amplitude * np.sin(freq * x))
#     fig.canvas.draw_idle()

# # Create interactive sliders
# interact(
#     update,
#     freq=FloatSlider(value=1.0, min=0.1, max=5.0, step=0.1, description='Frequency'),
#     amplitude=FloatSlider(value=1.0, min=0.1, max=2.0, step=0.1, description='Amplitude')
# )

def ait_plot(pixel_data, *, figsize=(12,6), fig=None, colorbar=True,
                label='counts/pixel', title=None,
                shrink=0.7, cmap='viridis', frame='galactic', log=True,
                value_format='auto', source_model=None, catalog=None,
                **kwargs):

    mp = pixel_data 
    if log: mp[mp==0] = np.nan

    afig = AITfigure(fig=fig, figsize=figsize, title=title)
    afig.imshow(mp, log=log,  cmap=cmap, **kwargs)
    _install_value_hover(afig.ax, getattr(afig, 'mappable', None), value_format=value_format)
    if colorbar:
        cbar = afig.colorbar(label=label, shrink=shrink)
        afig.figure._sky_display_colorbar = cbar
        _set_colorbar_visible(afig.ax, True)

    if catalog is None and source_model is not None:
        catalog = getattr(source_model, 'fermi_catalog', None)
    if catalog is not None and hasattr(catalog, 'select_cone'):
        center = SkyCoord(0.0, 0.0, unit='deg', frame='galactic')
        cone_size = 15.0
        catalog_subset = catalog.select_cone(center, cone_size=cone_size)
        if catalog_subset is not None and len(catalog_subset) > 0:
            if hasattr(catalog_subset, 'skycoord'):
                catalog_coords = catalog_subset.skycoord
            else:
                catalog_coords = SkyCoord(
                    catalog_subset.ra.values,
                    catalog_subset.dec.values,
                    unit='deg',
                    frame='fk5',
                )
            model_names = set() if source_model is None else {src.name for src in source_model}
            model_mask = catalog_subset.index.isin(model_names)
            for coord, name, in_model in zip(catalog_coords, catalog_subset.index, model_mask):
                sky = coord.transform_to('galactic')
                x = -np.radians(sky.l.deg)
                y = np.radians(sky.b.deg)
                afig.ax.text(
                    x, y,
                    str(name) if not str(name).startswith('FL16Y') else str(name)[5:],
                    color='red' if in_model else 'white',
                    fontsize=8,
                    ha='left',
                    va='bottom',
                )

    return afig

def zea_plot(center, pixel_data,  *, psf=None, figsize=(6, 5), r68=None,
        pixelsize=None, size=None, fig=None, axes_visible=True,
        cmap='viridis', colorbar=True, title=None, label='counts/pixel', log=True,
        vmin=None, vmax=None, frame='galactic', source_model=None,
        value_format='auto', catalog=None, **kwargs):

    # r68 gives the scale for size and pixel-size defaults if not provided.
    if r68 is not None:
        _size      = size      if size      is not None else 16 * r68
        _pixelsize = pixelsize if pixelsize is not None else r68 / 50
    else:
        _size      = size      if size      is not None else 5
        _pixelsize = pixelsize if pixelsize is not None else 0.05

    zfig = ZEAfigure(center, size=_size, fig=fig, figsize=figsize,frame=frame,
                        pixelsize=_pixelsize, axes_visible=axes_visible,
                        title='' if title is None else title)


    if log: pixel_data[pixel_data == 0] = np.nan
    zfig.imshow(pixel_data, log=log, #norm=LogNorm if log else Normalize, 
                 vmin=vmin, vmax=vmax, cmap=cmap, **kwargs)
    _install_value_hover(zfig.ax, getattr(zfig, 'mappable', None), value_format=value_format)
    if colorbar:
        cbar = zfig.colorbar(label=label, shrink=0.9, extend='max')
        zfig.figure._sky_display_colorbar = cbar
        _set_colorbar_visible(zfig.ax, True)
            #   color='white', ha='right', va='top', fontsize=12)

    # r68 PSF-size circle in lower left
    if r68 is not None:
        from matplotlib.patches import Circle
        ax = zfig.ax
        r68_px = r68 / _pixelsize
        cx, cy = (ax.transAxes + ax.transData.inverted()).transform((0.12, 0.12))
        ax.add_patch(Circle((cx, cy), r68_px,
                            fill=False, edgecolor='white', linewidth=1.5))
        ax.text(cx, cy, rf'${r68:.2f}^\circ$', color='white', fontsize=10,
                ha='center', va='center',)

    sm = source_model
    if catalog is None and sm is not None:
        catalog = getattr(sm, 'fermi_catalog', None)
    if catalog is not None and hasattr(catalog, 'select_cone'):
        cone_size = _size / np.sqrt(2.0)
        catalog_subset = catalog.select_cone(zfig.center, cone_size=cone_size)
        if catalog_subset is not None and len(catalog_subset) > 0:
            if hasattr(catalog_subset, 'skycoord'):
                catalog_coords = catalog_subset.skycoord
            else:
                catalog_coords = SkyCoord(
                    catalog_subset.ra.values,
                    catalog_subset.dec.values,
                    unit='deg',
                    frame='fk5',
                )

            model_names = set() if sm is None else {src.name for src in sm}
            model_mask = catalog_subset.index.isin(model_names)

            zfig.scatter(
                catalog_coords,
                marker='x',
                s=36,
                color='white',
                linewidths=0.8,
                alpha=0.8,
            )

            if np.any(model_mask):
                zfig.scatter(
                    catalog_coords[model_mask],
                    marker='o',
                    s=70,
                    facecolors='none',
                    edgecolors='red',
                    linewidths=1.5,
                )

            xpix, ypix = zfig.world_to_pixel(catalog_coords)
            nx, ny = zfig.array_shape
            hover_entries = []
            for x, y, name, in_model in zip(xpix, ypix, catalog_subset.index, model_mask):
                if not (0 <= x < nx and 0 <= y < ny):
                    continue
                text_artist = zfig.ax.text(
                    x + 4,
                    y + 4,
                    name if not str(name).startswith('FL16Y') else str(name)[5:],
                    color='red' if in_model else 'white',
                    fontsize=8,
                    ha='left',
                    va='bottom',
                )
                text_artist.set_picker(True)
                hover_entries.append(
                    (text_artist, _catalog_source_summary(catalog_subset.loc[name], in_model=in_model))
                )

            _install_text_hover(zfig.ax, hover_entries)

    return zfig


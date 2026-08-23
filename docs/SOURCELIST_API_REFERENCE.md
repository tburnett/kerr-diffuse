# Source List API Reference

**Module:** `like3/sourcelist.py`  
**Primary object:** `SourceModel`

This page is a quick developer reference for source-list operations used by the like3 likelihood workflow.

---

## Overview

`SourceModel` is a list-like container of source objects (typically `sources.PointSource`) with helper methods for:

- computing model counts, flux, and gradients,
- managing a flattened free-parameter view,
- selecting, adding, removing, and updating sources,
- creating temporary context-managed views for trial fitting/localization.

---

## Main Classes

## `SourceModel(list)`

### Constructor

```python
SourceModel(sources)
```

- `sources`: iterable of source objects.
- Builds internal parameter view (`self.parameters`) and resets selected-source state.

### Alternate constructor

```python
SourceModel.from_fermi_catalog(...)
```

- Loads a `utilities.catalogs.Fermi4FGL` catalog, or uses one passed via `catalog=`.
- Builds one `PointSource` per selected catalog row.
- Supports subset selection by:
  - exact source name or list of names via `select=...`
  - integer row indices or boolean masks via `select=...`
  - dataframe filtering via `query="..."`
  - spatial cone cuts via `skydir=(lon, lat)` and `cone_size=...`

### Core methods

- `model_counts(band, pix) -> np.ndarray`
  - Predicted counts for one band and pixel set.
  - Computes source flux at `band.energy`, applies PSF response, then exposure map.

- `flux(energies) -> np.ndarray`
  - Summed source-model flux across all sources.

- `gradient(energies) -> np.ndarray`
  - Gradient of total flux w.r.t active free parameters.

- `initialize(**kw)`
  - Rebuilds flattened parameter indexing using `parameterset.ParameterSet`.
  - Call after structural model changes (sources added/removed or model replaced).

- `parsubset(*select) -> parameterset.ParSubSet`
  - Returns a subset view of model parameters.

### Selection and source management

- `find_source(source_name) -> Source`
  - Accepts:
    - exact source name (string),
    - wildcard string (`"prefix*"` or `"*suffix"`),
    - source object,
    - `None` to reuse current selection.
  - Updates `selected_source` and `selected_source_index` on success.
  - Raises `SourceModelException` if selection fails.

- `add_source(newsource=None, **kw) -> Source | None`
  - Appends an existing source, or creates `sources.PointSource(**kw)` if `newsource` is `None`.
  - Rebuilds parameter view.
  - Returns `None` on duplicate source name.

- `del_source(source_name) -> Source`
  - Removes selected/matched source and rebuilds parameter view.

- `set_model(model, source_name=None) -> tuple`
  - Replaces one source model and returns `(source, old_model)`.
  - If `model` is a string, it is evaluated.
  - Rebuilds parameter view and refreshes default bounds.

- `list_sources()`
  - Prints all source entries in current order.

- `setposition(skydir)`
  - Sets position of the currently selected source.

### Context views

- `view() -> SourceModelContext`
  - Full snapshot/restore context.
  - Inside `with`, you may mutate source list and models.
  - On exit, restores list membership, per-source model state, and selection.

- `localization_view(source_name=None) -> SourceModelContext`
  - Position-only context for selected source.
  - `with` body receives `LocalizedSourceView`.
  - On exit, restores only selected source position.

- `localization_context(source_name=None)`
  - Backward-compatible alias to `localization_view`.

### Convenience properties

- `source_names`
- `models`
- `free`
- `bounds`
- `parameter_names`

---

## `SourceModelContext`

Context manager returned by `SourceModel.view()` and `SourceModel.localization_view()`.

### Behavior

- `__enter__`
  - captures snapshot.
  - returns `SourceModel` (full context) or `LocalizedSourceView` (position-only).

- `__exit__`
  - always restores snapshot (even on exception).
  - does not suppress exceptions.

---

## `LocalizedSourceView`

Helper object for localization workflows; delegates missing attributes to underlying `SourceModel`.

### Methods

- `delta_ts(loglike, position=None, baseline=None)`
  - Returns either:
    - one delta-TS value at `position`, or
    - a callable `f(position)` for repeated scans.
  - Uses:

$$
\Delta TS = 2\,(\log L(\text{trial}) - \log L_0)
$$

- Supports loglike callables of either form:
  - `loglike(position)`
  - `loglike()` (after temporary source-position assignment)

---

## Exceptions

- `SourceModelException`
  - Raised for source lookup/selection errors and invalid source-list operations.

---

## Usage Examples

### Build and inspect a demo model

```python
from like3.sourcelist import SourceModel

sl = SourceModel.demo(src_key=2)
print(sl)
sl.list_sources()
```

### Temporarily modify model state and auto-restore

```python
with sl.view() as trial:
    src = trial.find_source("Blazar")
    src.model.free[0] = False
    # run a trial fit / statistic here
# original source/model state restored
```

### Localization scan with position auto-restore

```python
with sl.localization_view("Blazar") as loc:
    f = loc.delta_ts(loglike_callable)
    ts = f(trial_position)
# selected source position restored
```

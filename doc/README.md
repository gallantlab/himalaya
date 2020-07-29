# Himalaya website

## Requirements

```
numpydoc
sphinx
sphinx_gallery
```

## Build the website

```bash
cd doc
make html
# ignore "WARNING: autosummary: stub file not found ..."
firefox _build/html/index.html
```

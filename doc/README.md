# Himalaya website

## Requirements

```
numpydoc
sphinx
sphinx_gallery
sphinxcontrib-mermaid
```

## Build the website

```bash
make html
# ignore "WARNING: autosummary: stub file not found ..."
firefox _build/html/index.html
```

## Push the website

```bash
make push-pages
```

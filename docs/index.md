# libssim

**libssim** is an open-source Python framework for forward modeling of Laser-Induced Breakdown Spectroscopy (LIBS) plasmas.

It draws inspiration from the work in:

> Herrera, K.K. (2008). *From Sample to Signal in Laser-Induced Breakdown Spectroscopy: An Experimental Assessment of Existing Algorithms and Theoretical Modeling Approaches*. PhD Dissertation, University of Florida.

## Features

- Immutable and validated `PlasmaState`
- Rich `Spectrum` container with metadata
- Strict SI units throughout
- Designed to support Monte Carlo LIBS-style optimization

## Quick Start

```python
from libssim.core import PlasmaState

state = PlasmaState(
    temperature_K=10000.0,
    electron_density_m3=1e23,
    total_density_m3=2e23,
    radius_m=1e-3,
    time_s=1e-6,
    composition={"Al": 0.95, "Mg": 0.05}
)

print(state)

Documentation
See the API Reference for full documentation.

---

### 3. `.github/workflows/docs.yml`

**Create the folders** `.github/workflows/` if they don’t exist, then create this file:

```yaml
name: Deploy Documentation

on:
  push:
    branches: [main]
  workflow_dispatch:

permissions:
  contents: read
  pages: write
  id-token: write

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"

      - name: Install dependencies
        run: pip install mkdocs mkdocstrings[python] mkdocs-material

      - name: Build site
        run: mkdocs build --clean

      - name: Upload artifact
        uses: actions/upload-pages-artifact@v3
        with:
          path: site/

  deploy:
    environment:
      name: github-pages
      url: ${{ steps.deployment.outputs.page_url }}
    runs-on: ubuntu-latest
    needs: build
    steps:
      - name: Deploy to GitHub Pages
        id: deployment
        uses: actions/deploy-pages@v4
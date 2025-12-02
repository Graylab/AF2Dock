# `AF2Dock`

Preprint: https://doi.org/10.1101/2025.11.28.691195

## Installation

First, install `openfold` following instructions on this page https://openfold.readthedocs.io/en/latest/Installation.html. Current repo uses this version of openfold https://github.com/aqlaboratory/openfold/releases/tag/v2.2.0.

(Notes from Dec 2025: I needed to remove `flash-attn` from the `environment.yml` file to successfully set up the enviroment.)

Then, in the enviroment with `openfold`, install AF2Dock as follows:

```
git clone https://github.com/Graylab/AF2Dock.git
cd AF2Dock
pip install .
```

If you run into issue installing `fastpdb` (required by `pinder`), follow instructions on the `pinder` repo page (https://github.com/pinder-org/pinder) to download the rust toolchain.

## Inference

Model weights are uploaded to Zenodo (https://doi.org/10.5281/zenodo.17782958). Detailed instructions coming soon.

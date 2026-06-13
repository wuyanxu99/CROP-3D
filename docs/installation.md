# Installation

The compact demos use standard scientific Python packages plus `plyfile`.

```bash
conda env create -f environment.yml
conda activate crop3d-paper
pip install -e .
```

For full reconstruction experiments, install the external tools separately:

- COLMAP for sparse reconstruction.
- CUDA-enabled PyTorch for training and rendering.
- 2DGS rasterization extensions.
- Depth Anything 3 weights, if running depth-prior generation.

The 2DGS CUDA extensions are included as source under `crop3d/reconstruction/submodules/` and can be installed with:

```bash
pip install crop3d/reconstruction/submodules/diff-surfel-rasterization
pip install crop3d/reconstruction/submodules/simple-knn
```

Those heavier assets are not bundled in this paper artifact.

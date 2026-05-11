



https://github.com/user-attachments/assets/70311b6e-9f32-4b9e-8007-4b362102b770

# GenoViewPython

GenoViewPython is a pure Python implementation of [GenoView](https://github.com/orangeduck/GenoView/) - a really basic example raylib application that can be used to view skeletal animation data in a way that is clear and highlights any artefacts. It uses a simple Deferred Renderer that supports shadow maps and Screen Space Ambient Occlusion, as well as a procedural grid shader as a texture. This makes common artefacts such as foot sliding and penetrations easy to see on a skinned character even on low-end devices, without the complexity of a full rendering engine.

Included are some simple maya scripts for exporting characters into a binary format that can be easily loaded by the application. These scripts are made for the Geno character from the following datasets:

* [LaFAN resolved](https://github.com/orangeduck/lafan1-resolved)
* [ZeroEGGS retargeted](https://github.com/orangeduck/zeroeggs-retarget)
* [Motorica retargeted](https://github.com/orangeduck/motorica-retarget)
* [100STYLE retargeted](https://github.com/orangeduck/100style-retarget)
* [InterAct retargeted](https://github.com/orangeduck/interact-retarget)

However they can likely be adapted to new characters, or the normal raylib-supported file formats can be loaded too.

# Getting Started

You need to first pip install the [python raylib bindings](https://electronstudio.github.io/raylib-python-cffi/):

```
pip install raylib
```

Then you should just be able to run `genoview.py`.

Here are the steps to viewing any of the animation data linked above in this viewer.

1. Download the BVH files for the animation dataset you want to view.
2. Place any bvh files you want to view in the `resources` folder.
3. Edit the line in `genoview.py` where `bvhData` is loaded to load the animation you want to view instead.



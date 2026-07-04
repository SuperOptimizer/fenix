# 0011 — render3d: VTK-hosted GigaVoxels volume engine, Metal beneath when needed

- Status: accepted (forrest, 2026-07-04: "yes gigavoxels exactly", "metal is approved",
  "leverage vtk as much as we can")
- Context: seeing-the-scroll needs real-time multiscale volumetric rendering of
  75784×32693×32693 @ 2.4 µm — instant zoom across scales, progressive refinement,
  eventual consistency. Stock VTK volume mappers require fully-resident vtkImageData;
  no out-of-core paging or per-ray LOD.
- Decision:
  1. VTK remains the HOST: scene, interaction, mesh PBR, HUD, picking, depth compositing.
  2. The volume core is a first-party GigaVoxels-lineage engine over the fxvol explicit
     LOD pyramid (brick = the native 64³ chunk, +1-voxel apron): brick atlas + page
     table + finest-resident-ancestor fallback (never a hole; top LODs pinned) +
     request-feedback streaming (view/prefetch priority-heap pattern, codec decode).
  3. Rung 1 = clipmap stack of stock vtkVolumes (nested LODs, async re-gather) — ships
     first, remains the far-field fallback. Rung 2 = custom mapper via
     vtkOpenGLGPUVolumeRayCastMapper shader replacements (atlas/page-table sampling,
     virtual-texturing feedback within macOS GL 4.1).
  4. Metal (via vendored metal-cpp, Apache-2.0 — approved) is the escalation path for
     the volume pass ONLY, rendering into an IOSurface-shared texture composited by the
     VTK scene, when GL 4.1 becomes the ceiling. Vulkan analog later for the Linux rig.
  5. Engine core stays API-agnostic C++ (u8-native, ZYX->XYZ swizzle once at upload,
     absent-vs-failed surfaced on screen, render thread never blocks on IO/decode).
- Consequences: maximum reuse (codec pyramid, block cache, prefetch heap, CachedVolume,
  view-surf mesh pass); portable to Linux from day one; Metal work is contained to one
  backend seam; the deprecated-GL risk on macOS is explicitly fenced by that same seam.

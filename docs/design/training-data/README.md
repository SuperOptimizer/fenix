# Surface-prediction training data — segment manifest

`segments-manifest.json` enumerates every ground-truth surface mesh in the
`vesuvius-challenge-open-data` S3 bucket (scanned 2026-07-02): all top-level scroll
prefixes that carry a `segments/` directory, and for each segment ONLY its
`mesh/<seg>-on-<volume>-<um>.tifxyz/` entries (x/y/z coordinate TIFFs + meta.json).
ink-detection/, surface-renders/, surface-volumes/, and mesh/intermediate/ are
deliberately excluded — the tifxyz mesh is the training GT; CT comes from the scroll's
own `volumes/` zarr at the matching volume id.

| scroll | segments | notes |
|---|---|---|
| PHercParis4 | 81 | 1.129/2.4/7.91/45.532 µm variants |
| PHerc0172 | 53 | 7.91 µm |
| PHerc0139 | 38 | 2.399 µm |
| PHerc1667 | 20 | 1.129/2.399/3.24/7.91 µm |
| PHerc1447 | 4 | unnamed `mesh/tifxyz/` variant (volume id via segment's surface-volume naming; um in meta.json scale) |

196 segments, 336 tifxyz meshes total. A segment with multiple meshes is the same
surface registered on different volumes/resolutions — pick per training volume.
Excluded: PHerc0009B (`segments/raw/` = loose PNGs, no meshes), PHerc1203 & PHerc1451
(1 segment each, no mesh/ directory).

Regenerate: scan `?list-type=2&delimiter=/` per prefix (see git history of this file
for the script, or ask the agent to re-run the enumeration).

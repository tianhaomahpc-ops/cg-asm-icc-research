#!/usr/bin/env python3
"""heart_torso.py -- conforming unstructured-tet mesh: 20x7x3 mm heart slab
embedded (centered) in a 50x50x50 mm torso box, for the coupled forward-ECG
FEM solve in forward_ecg.cpp.

WHY a single mesh with two tagged volumes:
  Gmsh OpenCASCADE BooleanFragments computes the common interface ONCE and
  deletes the duplicated face, so the heart and the torso share the exact
  same surface mesh (same nodes, same triangles) -> the heart-torso interface
  is CONFORMING by construction.  MFEM then reads the two physical volumes as
  domain attributes 1 (heart) / 2 (torso), and ParSubMesh::CreateFromDomain
  extracts the heart with a transfer map that lines up DOF-for-DOF on the
  shared interface.

Units: mm (matches monodomain.c: chi in /mm, sigma in mS/mm).

Output: heart_torso.msh  (Gmsh MSH 2.2 ASCII -- the ONLY format MFEM 4.9 reads).

Run:
  python3 heart_torso.py                 # needs the gmsh python module
  # or, if you only have the binary:
  gmsh - heart_torso.py                  # runs this as a gmsh script

Tunables (env or edit below): H_HEART (mm), H_TORSO (mm).
"""
import os
import sys
import gmsh

H_HEART = float(os.environ.get("H_HEART", "0.5"))   # target edge length in heart
H_TORSO = float(os.environ.get("H_TORSO", "3.0"))   # target edge length in torso
OUT     = os.environ.get("OUT", "heart_torso.msh")

# torso box (centered): [-25,25]^3 ;  heart slab (centered): x[-10,10] y[-3.5,3.5] z[-1.5,1.5]
TLX, TLY, TLZ = 50.0, 50.0, 50.0
HLX, HLY, HLZ = 20.0, 7.0, 3.0

gmsh.initialize()
gmsh.model.add("heart_torso")
occ = gmsh.model.occ

# 1) primitives ------------------------------------------------------------
torso = occ.addBox(-TLX/2, -TLY/2, -TLZ/2, TLX, TLY, TLZ)   # tag of the box volume
heart = occ.addBox(-HLX/2, -HLY/2, -HLZ/2, HLX, HLY, HLZ)

# 2) CONFORMING fragmentation (the load-bearing call) ----------------------
#    Splits the torso into (torso-minus-heart) and (heart) sharing ONE
#    interface surface; duplicate faces are removed -> conformal.
occ.fragment([(3, torso)], [(3, heart)])
occ.synchronize()

# 3) re-identify the two volumes by bounding box (BooleanFragments RENUMBERS
#    tags -- never assume the old tags survive).
vols = gmsh.model.getEntities(3)
heart_tag, torso_tag = None, None
for (dim, tag) in vols:
    xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.getBoundingBox(dim, tag)
    dx, dy, dz = xmax-xmin, ymax-ymin, zmax-zmin
    is_heart = (abs(dx-HLX) < 1e-6 and abs(dy-HLY) < 1e-6 and abs(dz-HLZ) < 1e-6)
    if is_heart:
        heart_tag = tag
    else:
        torso_tag = tag
assert heart_tag is not None and torso_tag is not None, "could not identify volumes"

# 4) surfaces:
#    - interface = boundary of the heart volume (heart is fully interior, so
#      all 6 slab faces are shared with the torso).
#    - body      = the 6 outer faces of the 50^3 box (centroid on |.|=25 plane).
iface_faces = [t for (d, t) in gmsh.model.getBoundary([(3, heart_tag)],
                                                       oriented=False, recursive=False)]
body_faces = []
for (d, t) in gmsh.model.getBoundary([(3, torso_tag)], oriented=False, recursive=False):
    if t in iface_faces:
        continue
    cx, cy, cz = gmsh.model.occ.getCenterOfMass(2, t)
    if (abs(abs(cx)-TLX/2) < 1e-6 or abs(abs(cy)-TLY/2) < 1e-6 or abs(abs(cz)-TLZ/2) < 1e-6):
        body_faces.append(t)

# 5) physical groups -> MFEM attributes (tags must be >= 1) -----------------
#    volumes -> DOMAIN attributes ; surfaces -> BOUNDARY attributes
gmsh.model.addPhysicalGroup(3, [heart_tag], tag=1, name="heart")     # domain attr 1
gmsh.model.addPhysicalGroup(3, [torso_tag], tag=2, name="torso")     # domain attr 2
gmsh.model.addPhysicalGroup(2, body_faces,  tag=1, name="body")      # bdr attr 1
gmsh.model.addPhysicalGroup(2, iface_faces, tag=2, name="interface") # bdr attr 2 (interior face)

# 6) graded mesh size: fine in/near the heart, coarse in the torso ---------
f = gmsh.model.mesh.field
f.add("Box", 1)
f.setNumber(1, "VIn",  H_HEART)
f.setNumber(1, "VOut", H_TORSO)
f.setNumber(1, "XMin", -HLX/2-2); f.setNumber(1, "XMax", HLX/2+2)
f.setNumber(1, "YMin", -HLY/2-2); f.setNumber(1, "YMax", HLY/2+2)
f.setNumber(1, "ZMin", -HLZ/2-2); f.setNumber(1, "ZMax", HLZ/2+2)
f.setNumber(1, "Thickness", 6.0)   # graded transition zone
f.setAsBackgroundMesh(1)
gmsh.option.setNumber("Mesh.MeshSizeMin", min(H_HEART, H_TORSO)*0.8)
gmsh.option.setNumber("Mesh.MeshSizeMax", H_TORSO*1.3)
gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)

# 3D Delaunay -> genuinely unstructured tets (NOT a Cartesian split)
gmsh.option.setNumber("Mesh.Algorithm", 5)     # 2D Delaunay
gmsh.option.setNumber("Mesh.Algorithm3D", 1)   # 3D Delaunay (use 10=HXT for speed)
gmsh.model.mesh.generate(3)

# 7) export MSH 2.2 ASCII (MFEM 4.9 requirement) ---------------------------
gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
gmsh.option.setNumber("Mesh.Binary", 0)
gmsh.option.setNumber("Mesh.SaveAll", 0)       # keep only physical-group elements
gmsh.write(OUT)

# small report
ntet = len(gmsh.model.mesh.getElementsByType(4)[0])
print(f"[heart_torso] heart vol tag={heart_tag} torso vol tag={torso_tag}")
print(f"[heart_torso] interface faces={len(iface_faces)} body faces={len(body_faces)}")
print(f"[heart_torso] tets={ntet}  wrote {OUT} (MSH 2.2 ASCII)")
print(f"[heart_torso] domain attrs: 1=heart 2=torso ; bdr attrs: 1=body 2=interface")
if "-nopopup" not in sys.argv and os.environ.get("GMSH_GUI"):
    gmsh.fltk.run()
gmsh.finalize()

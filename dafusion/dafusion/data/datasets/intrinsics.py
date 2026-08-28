"""Per-dataset camera intrinsics (fx, fy, cx, cy) for HHA depth encoding.

HHA (Gupta et al.) needs a camera matrix K to back-project depth to a point cloud
(for the height-above-ground and surface-normal channels). None ship with the data,
so these are documented best-effort values and a KNOWN reproduction risk (see
dafusion/README.md). Override per run with MODEL / dataset config if exact values
are recovered from the UOAIS-Sim BlenderProc setup or the sensor calibrations.

Resolutions are NOT uniform: OCID/OSD/UOAIS-Sim/TOD are 640x480, but OCBD is 600x400. Values below
are quoted at each entry's own native resolution, so rescale by (w_eval/w_native) before use if the
image is resized.
"""

# (fx, fy, cx, cy)
INTRINSICS = {
    # UOAIS-Sim: BlenderProc render, 640x480. Value assumed (~60 deg HFOV pinhole);
    # VERIFY against the UOAIS-Sim generation config.
    "uoais_sim": (591.0125, 590.16775, 320.0, 240.0),
    # OSD: Kinect v1 (PrimeSense), 640x480. STILL ASSUMED -- this copy of OSD ("OSD-0.20-depth") ships
    # only disparity PNGs, not the ./pcd/ clouds its ReadMe describes, so there is nothing to fit
    # against. Same lab (ACIN) and sensor family as OCID, so 570.34 may well apply here too, but that
    # is an inference, not a measurement. Do not "fix" it by picking whichever value scores better.
    "osd": (525.0, 525.0, 319.5, 239.5),
    # OCID: RECOVERED, not assumed. Least-squares fit of X/Z against u (and Y/Z against v) over 60 of
    # OCID's own organized pcd clouds, sampled across 60 different sequences, gives
    # fx = fy = 570.34 +- 0.01, cx = 319.50, cy = 239.50. The std of 0.01 px means the shipped cloud is
    # a self-consistent pinhole reprojection, so this is the dataset's real geometry rather than a fit to
    # anything we care about scoring. Implies HFOV 58.6 / VFOV 45.6 deg, which matches the ASUS Xtion Pro
    # Live spec (58 x 45) and independently corroborates the number.
    # The previous (525, 525) was the textbook Kinect-v1 value and was 8.6% short in focal length; the
    # principal point was already correct. Consumed by the xyz / depth_normals back-projection AND by
    # benchmark.pixel_area_cm2, so the cm^2 size gate was computing areas ~15% off on OCID.
    # Recover with: python scripts/recover_intrinsics.py --dataset ocid
    "ocid": (570.34, 570.34, 319.5, 239.5),
    # OCBD: 600x400 (NOT 640x480 -- its annotations and organized cloud are both 600x400).
    # RECOVERED, not assumed: least-squares fit of X=(u-cx)z/fx over 60 of OCBD's own organized
    # metric clouds gives fx 617.71+-0.33, fy 616.15+-0.29, cx 299.24, cy 204.16 (std across
    # clouds < 0.4 px, i.e. the cloud is self-consistent to well under a pixel). Implies
    # HFOV 51.8 / VFOV 36.0 deg. The previous (503, 503, 320, 240) was a guess from the paper's
    # "Azure Kinect DK" and was ~19% off in focal length and 36 px off in cy.
    # Only the HHA path consumes this for OCBD; xyz/depth_normals use the metric cloud directly.
    "ocbd": (617.71, 616.15, 299.24, 204.16),
    # TOD (tabletop_dataset_v5): 640x480, 45deg vertical FOV -> fx=fy~=579.41 (UCN/MSMFormer).
    "tabletop": (579.41, 579.41, 320.0, 240.0),
}

DEFAULT = INTRINSICS["uoais_sim"]


def get_intrinsics(name):
    return INTRINSICS.get(name, DEFAULT)

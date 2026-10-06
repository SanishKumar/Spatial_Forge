# Where this sits

Short version: nothing in SpatialForge is a new capability. It is a careful
reimplementation of a well-understood pipeline, built to a different
standard of evidence than the systems it reimplements. If you need a
reconstruction, use one of those.

## The lineage

- **Volumetric fusion.** Averaging truncated signed distances from many
  range images into one grid, and extracting the zero level set, is
  [Curless and Levoy, 1996](https://doi.org/10.1145/237170.237269). Every
  system below, including this one, is that idea.
- **Doing it live.**
  [KinectFusion](https://doi.org/10.1109/ISMAR.2011.6092378) (Newcombe et
  al., 2011) fused a moving depth camera into a dense grid on a GPU in real
  time, tracking the camera against the model as it went.
- **Doing it sparsely.** A dense grid wastes almost all of its memory on
  empty space.
  [Voxel hashing](https://niessnerlab.org/projects/niessner2013hashing.html)
  (Nießner et al., 2013) stores small voxel blocks only near surfaces and
  finds them through a hash table. The 8×8×8 blocks here are that design.
- **Mature implementations.**
  [InfiniTAM](https://arxiv.org/abs/1410.0925),
  [voxblox](https://arxiv.org/abs/1611.03631),
  [VDBFusion](https://pmc.ncbi.nlm.nih.gov/articles/PMC8838740/),
  [nvblox](https://arxiv.org/abs/2311.00626) and Open3D's voxel block grid.
  All are faster than this by orders of magnitude, and several run in real
  time on embedded hardware.

## What is different here

Those systems are built for speed and are, reasonably, willing to pay for it
in exact repeatability. The InfiniTAM report is candid about one such
trade. Describing block allocation, it says the aim was "to minimise the use
of blocking operations (e.g. atomics) and to completely avoid the use of
critical sections", that with non-atomic writes "if more than one new block
has to be allocated with the same hash index, only the most recently written
allocation will actually be performed", and that "we tolerate such artefacts
from intra-frame hash collisions, as they will be corrected in the next
frame automatically".

That is the right decision for a robot that needs a map now. It also means
the stored volume is the outcome of a race, not a function of the input.

The same is true further up the stack. The ORB-SLAM2 paper
([Mur-Artal and Tardós, 2017](https://arxiv.org/abs/1610.06475)) evaluates
by running "each sequence 5 times" and reporting medians, "to account for
the non-deterministic nature of the multi-threading system". Reporting a
median over runs is the field's normal, honest answer to a system whose
output varies.

SpatialForge takes the other side of that trade. It is single-threaded and
slow, and in exchange:

- the same scan produces the same bytes — for the committed fixture, CI
  checks the volume and mesh digests on three operating systems and two
  interpreter versions;
- every artifact names, by digest, the artifact it was made from;
- every optimised path is required to match a slower reference exactly,
  rather than to within a tolerance.

The last of these runs into a well-known problem. Floating-point addition is
not associative, so reordering a sum changes its last bits, and vectorising
or parallelising a computation reorders sums. Numerical libraries treat
bitwise reproducibility as a research topic in its own right — see Demmel
and Nguyen's work on
[reproducible summation](https://www2.eecs.berkeley.edu/Pubs/TechRpts/2016/EECS-2016-121.html).
The approach here is the unsophisticated one: never reorder. Each
vectorised path performs the reference's operations in the reference's
order, and a test holds it to that.

Whether that trade is worth making depends entirely on the use. For
real-time robotics it is not. For a map that someone will be navigated
through, where being able to say exactly what was built from what matters
more than building it quickly, it might be.

## What it does not attempt

**Pose estimation.** This pipeline is told where the camera was. Recovering
that is the harder half of the problem and is a field of its own, from
feature-based SLAM such as ORB-SLAM to the recent feed-forward models that
predict geometry and cameras directly from images:
[DUSt3R](https://arxiv.org/abs/2312.14132),
[MASt3R](https://arxiv.org/abs/2406.09756),
[VGGT](https://arxiv.org/abs/2503.11651), and streaming systems built on
them such as [LingBot-Map](https://github.com/Robbyant/lingbot-map). Those
take ordinary video, with no depth sensor and no known trajectory, which
this cannot.

**Absolute accuracy.** The validation here measures agreement between
viewpoints. The standard benchmark for surface accuracy is
[ICL-NUIM](https://www.doc.ic.ac.uk/~ahanda/VaFRIC/iclnuim.html), a
synthetic scene with a ground-truth model, where published systems report
mean distances of roughly a centimetre with *estimated* poses. Running it is
the obvious next piece of evidence and has not been done.

**Relocalization, semantics, real time.** None of them.

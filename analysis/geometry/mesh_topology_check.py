"""Phase 1, Option 0: does the mesh generator preserve topology under a change of (d, s)?

The whole transfer construction of Chapter 2's conclusions -- a reference mesh,
a closed-form map varphi, the Piola pull-back, a Helmholtz projection -- exists
only because a change of geometry is assumed to *regenerate* the mesh, so that
the snapshot matrix cannot be formed.  If instead the generator produces the
same connectivity and the same node numbering, and moves only the node
coordinates, then a proper orthogonal decomposition across geometries is
immediately well defined and none of that machinery is needed.

The check is one meshing job per geometry.  This script compares the resulting
Ensight geometry files: node count, per-part element counts, and the
connectivity itself.  Coordinates are expected to differ; nothing else should.

Usage:  mesh_topology_check.py base_dir variant_dir [variant_dir ...]
"""
import hashlib
import sys


def parse_geom(path):
    """Return (n_nodes, coord_block_lines, topology_text) for an alucell
    formatted Ensight .geom file."""
    with open(path) as f:
        lines = f.read().split("\n")
    i = next(k for k, l in enumerate(lines) if l.strip() == "coordinates")
    n = int(lines[i + 1])
    coords = lines[i + 2: i + 2 + n]
    topo = "\n".join(lines[i + 2 + n:])
    return n, coords, topo


def part_summary(topo):
    """Element counts per (part, element type), read from the topology text."""
    out, part, etype = [], None, None
    it = iter(topo.split("\n"))
    for l in it:
        s = l.strip()
        if s.startswith("part"):
            part = s
            etype = None
        elif s and not s[0].isdigit() and not s[0] == "-" and part and " " not in s:
            etype = s
            try:
                cnt = int(next(it).strip())
                out.append((part, etype, cnt))
            except (StopIteration, ValueError):
                pass
    return out


def main(dirs):
    ref = None
    print("%-12s %10s %12s %-34s %s"
          % ("variant", "nodes", "topo bytes", "topology sha1 (first 16)", "verdict"))
    print("-" * 92)
    for d in dirs:
        p = "%s/ensight/ensight_mesh.geom" % d.rstrip("/")
        n, coords, topo = parse_geom(p)
        h = hashlib.sha1(topo.encode()).hexdigest()
        name = d.rstrip("/").split("/")[-1]
        if ref is None:
            ref = (n, h, topo, coords)
            verdict = "(reference)"
        else:
            same_n = n == ref[0]
            same_t = h == ref[1]
            moved = sum(1 for a, b in zip(coords, ref[3]) if a != b)
            verdict = ("TOPOLOGY IDENTICAL, %d/%d nodes moved" % (moved, n)
                       if same_n and same_t else
                       "DIFFERS (nodes %s, topology %s)"
                       % ("same" if same_n else "differ",
                          "same" if same_t else "differ"))
        print("%-12s %10d %12d %-34s %s" % (name, n, len(topo), h[:16], verdict))

    print()
    ps = part_summary(ref[2])
    if ps:
        print("element blocks in the reference mesh:")
        for part, et, cnt in ps[:20]:
            print("   %-24s %-10s %9d" % (part, et, cnt))


if __name__ == "__main__":
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    main(sys.argv[1:])

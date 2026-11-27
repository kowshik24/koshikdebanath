#!/usr/bin/env python3
"""
Generate the site's editorial SVG illustrations ("house medium").

House medium (see assets/img/art/ART_DIRECTION.md):
  ground   warm cream cotton paper #FAF9F5 + fine grain
  linework charcoal ink #141413, slightly wobbly (hand-drawn)
  accents  terracotta #D97757 / earth #C6613F, slate blue #6A9BCC, sage #788C5D
  paper    sand #E3DACC cut-outs with torn edges
  pencil   #B0AEA5 / #C6C4BA faint guides

Each piece anchors on ONE mechanism with generous negative space.
Seeded RNG -> deterministic output. Run:  python3 tools/generate_art.py
"""
import math
import os
import random

OUT = os.path.join(os.path.dirname(__file__), "..", "assets", "img", "art")
W, H = 800, 500

PAPER = "#FAF9F5"
INK = "#141413"
CLAY = "#D97757"
EARTH = "#C6613F"
SLATE = "#6A9BCC"
SAGE = "#788C5D"
SAND = "#E3DACC"
PENCIL = "#B0AEA5"
PENCIL_LIGHT = "#C6C4BA"
GRAY = "#87867F"


def f(v):
    return f"{v:.1f}"


def defs(seed):
    return f"""
  <defs>
    <!-- paper grain -->
    <filter id="grain" x="0" y="0" width="100%" height="100%">
      <feTurbulence type="fractalNoise" baseFrequency="0.85" numOctaves="2" seed="{seed}" result="n"/>
      <feColorMatrix in="n" type="matrix" values="0 0 0 0 0.08  0 0 0 0 0.08  0 0 0 0 0.07  0 0 0 0.05 0"/>
    </filter>
    <!-- hand-drawn ink wobble -->
    <filter id="ink" x="-5%" y="-5%" width="110%" height="110%">
      <feTurbulence type="fractalNoise" baseFrequency="0.035" numOctaves="2" seed="{seed + 3}" result="w"/>
      <feDisplacementMap in="SourceGraphic" in2="w" scale="2.6" xChannelSelector="R" yChannelSelector="G"/>
    </filter>
    <!-- torn / hand-cut paper edge -->
    <filter id="torn" x="-10%" y="-10%" width="120%" height="120%">
      <feTurbulence type="fractalNoise" baseFrequency="0.18" numOctaves="3" seed="{seed + 7}" result="t"/>
      <feDisplacementMap in="SourceGraphic" in2="t" scale="2.2" xChannelSelector="R" yChannelSelector="G"/>
    </filter>
    <!-- soft charcoal shadow -->
    <filter id="soot" x="-20%" y="-20%" width="140%" height="140%">
      <feGaussianBlur stdDeviation="2.2"/>
    </filter>
    <!-- graphite crosshatch -->
    <pattern id="hatch" width="6" height="6" patternUnits="userSpaceOnUse" patternTransform="rotate(38)">
      <line x1="0" y1="0" x2="0" y2="6" stroke="{INK}" stroke-width="0.7" opacity="0.55"/>
    </pattern>
    <pattern id="crosshatch" width="6" height="6" patternUnits="userSpaceOnUse" patternTransform="rotate(-30)">
      <line x1="0" y1="0" x2="0" y2="6" stroke="{INK}" stroke-width="0.6" opacity="0.5"/>
      <line x1="0" y1="0" x2="6" y2="0" stroke="{INK}" stroke-width="0.6" opacity="0.35"/>
    </pattern>
  </defs>"""


def svg(name, title, body, seed):
    doc = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" role="img" aria-labelledby="t">
  <title id="t">{title}</title>{defs(seed)}
  <rect width="{W}" height="{H}" fill="{PAPER}"/>
{body}
  <rect width="{W}" height="{H}" filter="url(#grain)"/>
</svg>
"""
    path = os.path.join(OUT, name)
    with open(path, "w") as fh:
        fh.write(doc)
    print("wrote", os.path.relpath(path))


def catmull(points, closed=False):
    """Smooth path through points (Catmull-Rom -> cubic Bezier)."""
    pts = points[:]
    if closed:
        pts = [pts[-1]] + pts + [pts[0], pts[1]]
    else:
        pts = [pts[0]] + pts + [pts[-1]]
    d = f"M{f(pts[1][0])},{f(pts[1][1])}"
    for i in range(1, len(pts) - 2):
        p0, p1, p2, p3 = pts[i - 1], pts[i], pts[i + 1], pts[i + 2]
        c1 = (p1[0] + (p2[0] - p0[0]) / 6, p1[1] + (p2[1] - p0[1]) / 6)
        c2 = (p2[0] - (p3[0] - p1[0]) / 6, p2[1] - (p3[1] - p1[1]) / 6)
        d += f" C{f(c1[0])},{f(c1[1])} {f(c2[0])},{f(c2[1])} {f(p2[0])},{f(p2[1])}"
    if closed:
        d += " Z"
    return d


def scrap(rng, cx, cy, r, n=None):
    """Irregular paper-scrap polygon."""
    n = n or rng.randint(5, 7)
    pts = []
    a0 = rng.uniform(0, math.tau)
    for i in range(n):
        a = a0 + i * math.tau / n + rng.uniform(-0.25, 0.25)
        rr = r * rng.uniform(0.7, 1.15)
        pts.append(f"{f(cx + rr * math.cos(a))},{f(cy + rr * math.sin(a))}")
    return " ".join(pts)


# ---------------------------------------------------------------------------
# 1. Compression / latent representations  (Machine Learning)
# ---------------------------------------------------------------------------
def compression():
    rng = random.Random(11)
    out = []
    # hand-ruled alignment lines (right side)
    out.append(f'  <g stroke="{PENCIL_LIGHT}" stroke-width="0.8" filter="url(#ink)">')
    for y in (232, 268):
        out.append(f'    <line x1="430" y1="{y}" x2="730" y2="{y + rng.uniform(-1, 1):.1f}"/>')
    out.append(f'    <line x1="430" y1="250" x2="730" y2="250" stroke-dasharray="2 6"/>')
    out.append("  </g>")

    # tangle of threads funnelling into prism
    out.append(f'  <g fill="none" stroke="{INK}" stroke-linecap="round" filter="url(#ink)">')
    for i in range(46):
        pts = []
        x, y = rng.uniform(80, 220), rng.uniform(130, 370)
        pts.append((x, y))
        for _ in range(rng.randint(3, 5)):
            x = min(max(x + rng.uniform(-60, 70), 70), 320)
            y = min(max(y + rng.uniform(-80, 80), 110), 390)
            pts.append((x, y))
        # converge
        pts.append((rng.uniform(330, 355), 250 + rng.uniform(-40, 40)))
        pts.append((392, 250 + rng.uniform(-14, 14)))
        op = rng.uniform(0.35, 0.85)
        sw = rng.uniform(0.5, 1.1)
        out.append(f'    <path d="{catmull(pts)}" stroke-width="{sw:.2f}" opacity="{op:.2f}"/>')
    out.append("  </g>")

    # prism (narrow vertical cut-paper) with graphite shading
    out.append('  <g filter="url(#torn)">')
    out.append(f'    <polygon points="388,150 412,166 412,334 388,350" fill="{SAND}"/>')
    out.append('    <polygon points="400,158 412,166 412,334 400,342" fill="url(#hatch)"/>')
    out.append("  </g>")
    out.append(f'  <polygon points="388,150 412,166 412,334 388,350" fill="none" stroke="{INK}" stroke-width="1.3" filter="url(#ink)"/>')

    # ordered row of tiles
    out.append('  <g filter="url(#torn)">')
    colors = [CLAY, PAPER, CLAY, SAND, PAPER, CLAY, SAND, PAPER]
    for i, c in enumerate(colors):
        x = 446 + i * 34
        out.append(f'    <rect x="{x}" y="236" width="26" height="28" fill="{c}" stroke="{INK}" stroke-width="1"/>')
    out.append("  </g>")
    # short exit strokes from prism to tiles
    out.append(f'  <g stroke="{INK}" stroke-width="0.9" filter="url(#ink)" opacity="0.7">')
    for dy in (-6, 0, 6):
        out.append(f'    <line x1="414" y1="{250 + dy}" x2="438" y2="{250 + dy * 0.4:.1f}"/>')
    out.append("  </g>")
    svg("compression.svg", "Tangled ink threads compressed through a narrow prism into an ordered row of tiles", "\n".join(out), 11)


# ---------------------------------------------------------------------------
# 2. Interpretability / circuits / tracing a path
# ---------------------------------------------------------------------------
def interpretability():
    rng = random.Random(23)
    out = []
    cols, rows = 6, 5
    x0, x1, y0, y1 = 170, 630, 130, 370
    xs = [x0 + i * (x1 - x0) / (cols - 1) for i in range(cols)]
    ys = [y0 + j * (y1 - y0) / (rows - 1) for j in range(rows)]
    nodes = [[(x + rng.uniform(-6, 6), y + rng.uniform(-6, 6)) for y in ys] for x in xs]

    # layered paper sheets behind each column
    out.append('  <g filter="url(#torn)" opacity="0.55">')
    for x in xs:
        out.append(f'    <rect x="{f(x - 16)}" y="{y0 - 34}" width="32" height="{y1 - y0 + 68}" fill="{SAND}"/>')
    out.append("  </g>")

    path_rows = [3, 2, 2, 1, 2, 1]

    # faint pencil branches
    out.append(f'  <g stroke="{PENCIL}" stroke-width="0.6" fill="none" filter="url(#ink)">')
    for i in range(cols - 1):
        for j in range(rows):
            for k in range(rows):
                if abs(j - k) > 2 or rng.random() < 0.35:
                    continue
                a, b = nodes[i][j], nodes[i + 1][k]
                op = rng.uniform(0.25, 0.6)
                out.append(f'    <line x1="{f(a[0])}" y1="{f(a[1])}" x2="{f(b[0])}" y2="{f(b[1])}" opacity="{op:.2f}"/>')
    out.append("  </g>")

    # highlighted continuous path
    pts = [nodes[i][path_rows[i]] for i in range(cols)]
    lead_in = (x0 - 70, pts[0][1] + 18)
    lead_out = (x1 + 70, pts[-1][1] - 12)
    d = catmull([lead_in] + pts + [lead_out])
    out.append(f'  <path d="{d}" fill="none" stroke="{CLAY}" stroke-width="3.2" stroke-linecap="round" filter="url(#ink)"/>')

    # nodes
    out.append('  <g filter="url(#ink)">')
    for i in range(cols):
        for j in range(rows):
            x, y = nodes[i][j]
            if path_rows[i] == j:
                out.append(f'    <circle cx="{f(x)}" cy="{f(y)}" r="8" fill="{CLAY}" stroke="{INK}" stroke-width="1.2"/>')
            else:
                fill = rng.choice([PAPER, PAPER, PENCIL_LIGHT])
                out.append(f'    <circle cx="{f(x)}" cy="{f(y)}" r="6" fill="{fill}" stroke="{GRAY}" stroke-width="0.9"/>')
    out.append("  </g>")
    svg("interpretability.svg", "A layered grid of nodes with one continuous path traced in terracotta ink", "\n".join(out), 23)


# ---------------------------------------------------------------------------
# 3. Evaluation / benchmarks / hidden failure modes
# ---------------------------------------------------------------------------
def evaluation():
    rng = random.Random(37)
    out = []
    cols, rows, s, gap = 12, 6, 26, 8
    gw = cols * s + (cols - 1) * gap
    gh = rows * s + (rows - 1) * gap
    ox, oy = (W - gw) / 2, (H - gh) / 2
    tilted = {(3, 1): 11, (8, 3): -13, (10, 0): 8}
    accents = {(1, 4): SLATE, (6, 2): CLAY, (5, 5): SLATE, (11, 4): CLAY}

    # ledger ruling
    out.append(f'  <g stroke="{PENCIL_LIGHT}" stroke-width="0.7" filter="url(#ink)">')
    out.append(f'    <line x1="{f(ox - 30)}" y1="{f(oy - 18)}" x2="{f(ox + gw + 30)}" y2="{f(oy - 18)}"/>')
    out.append(f'    <line x1="{f(ox - 30)}" y1="{f(oy + gh + 18)}" x2="{f(ox + gw + 30)}" y2="{f(oy + gh + 18)}"/>')
    out.append(f'    <line x1="{f(ox - 16)}" y1="{f(oy - 30)}" x2="{f(ox - 16)}" y2="{f(oy + gh + 30)}"/>')
    out.append("  </g>")

    tiles, lifted = [], []
    for i in range(cols):
        for j in range(rows):
            x, y = ox + i * (s + gap), oy + j * (s + gap)
            if (i, j) in tilted:
                lifted.append((x, y, tilted[(i, j)], accents.get((i, j), PAPER)))
            else:
                tiles.append((x, y, accents.get((i, j), PAPER)))

    # charcoal hollows revealed under tilted tiles
    for x, y, a, _ in lifted:
        out.append(f'  <rect x="{f(x + 2)}" y="{f(y + 3)}" width="{s}" height="{s}" fill="{INK}" opacity="0.55" filter="url(#soot)"/>')
        out.append(f'  <rect x="{f(x)}" y="{f(y)}" width="{s}" height="{s}" fill="url(#crosshatch)" filter="url(#ink)"/>')

    out.append('  <g filter="url(#torn)">')
    for x, y, c in tiles:
        out.append(f'    <rect x="{f(x)}" y="{f(y)}" width="{s}" height="{s}" fill="{c}" stroke="{INK}" stroke-width="0.9"/>')
    for x, y, a, c in lifted:
        cx, cy = x + s / 2, y + s / 2
        out.append(f'    <rect x="{f(x - 3)}" y="{f(y - 6)}" width="{s}" height="{s}" fill="{c}" stroke="{INK}" stroke-width="1" transform="rotate({a} {f(cx)} {f(cy)})"/>')
    out.append("  </g>")
    svg("evaluation.svg", "An orderly ledger of paper tiles where a few tilt up to reveal dark charcoal hollows beneath", "\n".join(out), 37)


# ---------------------------------------------------------------------------
# 4. Steering / alignment / controlled intervention
# ---------------------------------------------------------------------------
def steering():
    rng = random.Random(41)
    out = []
    # dotted arc
    cx, cy, r = 400, 520, 330
    arc = []
    for k in range(60):
        a = math.radians(205 + k * (130 / 59))
        arc.append((cx + r * math.cos(a), cy + r * math.sin(a)))
    out.append(f'  <path d="{catmull(arc)}" fill="none" stroke="{GRAY}" stroke-width="1.6" stroke-dasharray="1.5 9" stroke-linecap="round" filter="url(#ink)"/>')

    # cluster of paper circles on the arc, slightly ahead of where the pointer touches
    out.append('  <g filter="url(#torn)">')
    base_a = math.radians(272)
    circles = []
    for k in range(9):
        a = base_a + k * 0.034 + rng.uniform(-0.012, 0.012)
        rr = r + rng.uniform(-12, 12)
        x, y = cx + rr * math.cos(a), cy + rr * math.sin(a)
        rad = rng.uniform(9, 16)
        col = CLAY if k in (0, 4) else (EARTH if k == 7 else rng.choice([PAPER, SAND]))
        circles.append((x, y, rad))
        out.append(f'    <circle cx="{f(x)}" cy="{f(y)}" r="{f(rad)}" fill="{col}" stroke="{INK}" stroke-width="1"/>')
    out.append("  </g>")

    # fulcrum + lever / drafting pointer
    leftmost = min(circles, key=lambda c: c[0] - c[2])
    tip = (leftmost[0] - leftmost[2] - 4, leftmost[1] + 4)
    pivot = (230, 330)
    tail = (pivot[0] - (tip[0] - pivot[0]) * 0.55, pivot[1] - (tip[1] - pivot[1]) * 0.55)
    out.append('  <g filter="url(#ink)">')
    out.append(f'    <polygon points="{pivot[0]},{pivot[1] + 2} {pivot[0] - 18},{pivot[1] + 36} {pivot[0] + 18},{pivot[1] + 36}" fill="{SAND}" stroke="{INK}" stroke-width="1.3"/>')
    out.append(f'    <polygon points="{pivot[0]},{pivot[1] + 12} {pivot[0] - 11},{pivot[1] + 36} {pivot[0] + 11},{pivot[1] + 36}" fill="url(#hatch)"/>')
    out.append(f'    <line x1="{pivot[0] - 60}" y1="{pivot[1] + 37}" x2="{pivot[0] + 60}" y2="{pivot[1] + 37}" stroke="{INK}" stroke-width="1.2"/>')
    out.append(f'    <line x1="{f(tail[0])}" y1="{f(tail[1])}" x2="{f(tip[0])}" y2="{f(tip[1])}" stroke="{INK}" stroke-width="2.2" stroke-linecap="round"/>')
    out.append(f'    <circle cx="{pivot[0]}" cy="{pivot[1]}" r="3.5" fill="{PAPER}" stroke="{INK}" stroke-width="1.3"/>')
    out.append(f'    <circle cx="{f(tail[0])}" cy="{f(tail[1])}" r="7" fill="{EARTH}" stroke="{INK}" stroke-width="1.2"/>')
    out.append("  </g>")
    # tiny motion strokes
    out.append(f'  <g stroke="{CLAY}" stroke-width="1.4" stroke-linecap="round" filter="url(#ink)">')
    for k in range(3):
        y = tip[1] - 14 + k * 9
        out.append(f'    <line x1="{f(tip[0] - 30)}" y1="{f(y)}" x2="{f(tip[0] - 14)}" y2="{f(y - 2)}"/>')
    out.append("  </g>")
    svg("steering.svg", "A drafting lever gently nudging a cluster of paper circles along a dotted arc", "\n".join(out), 41)


# ---------------------------------------------------------------------------
# 5. Retrieval / search / semantic space  (NLP)
# ---------------------------------------------------------------------------
def retrieval():
    rng = random.Random(53)
    out = []
    gx0, gy0, gx1, gy1 = 150, 100, 680, 400
    # quiet coordinate grid
    out.append(f'  <g stroke="{PENCIL_LIGHT}" stroke-width="0.6" filter="url(#ink)">')
    for x in range(gx0, gx1 + 1, 44):
        out.append(f'    <line x1="{x}" y1="{gy0}" x2="{x}" y2="{gy1}" stroke-dasharray="1 5"/>')
    for y in range(gy0, gy1 + 1, 44):
        out.append(f'    <line x1="{gx0}" y1="{y}" x2="{gx1}" y2="{y}" stroke-dasharray="1 5"/>')
    out.append("  </g>")
    # axes
    out.append(f'  <g stroke="{INK}" stroke-width="1.3" fill="none" stroke-linecap="round" filter="url(#ink)">')
    out.append(f'    <path d="M{gx0},{gy0 - 10} L{gx0},{gy1 + 6} L{gx1 + 14},{gy1 + 6}"/>')
    out.append(f'    <path d="M{gx0 - 5},{gy0 - 2} L{gx0},{gy0 - 12} L{gx0 + 5},{gy0 - 2}"/>')
    out.append(f'    <path d="M{gx1 + 6},{gy1 + 1} L{gx1 + 16},{gy1 + 6} L{gx1 + 6},{gy1 + 11}"/>')
    out.append("  </g>")

    islands = [((560, 170), CLAY), ((600, 320), SAGE), ((440, 260), SAND)]
    # clustered islands
    out.append('  <g filter="url(#torn)">')
    for (ix, iy), col in islands:
        for k in range(7):
            a = rng.uniform(0, math.tau)
            d = rng.uniform(0, 22)
            out.append(f'    <polygon points="{scrap(rng, ix + d * math.cos(a), iy + d * math.sin(a), rng.uniform(7, 12))}" fill="{col}" stroke="{INK}" stroke-width="0.9"/>')
    out.append("  </g>")

    # drifting fragments + dotted trails
    out.append(f'  <g fill="none" stroke="{GRAY}" stroke-width="1" stroke-dasharray="1.5 6" stroke-linecap="round" filter="url(#ink)">')
    frags = []
    for k in range(9):
        sx, sy = rng.uniform(175, 360), rng.uniform(115, 385)
        (ix, iy), col = islands[k % len(islands)]
        t = rng.uniform(0.25, 0.5)
        mx, my = sx + (ix - sx) * t, sy + (iy - sy) * t + rng.uniform(-25, 25)
        frags.append((sx, sy, col))
        out.append(f'    <path d="{catmull([(sx, sy), (mx, my), (ix - (ix - sx) * 0.2, iy - (iy - sy) * 0.2)])}"/>')
    out.append("  </g>")
    out.append('  <g filter="url(#torn)">')
    for sx, sy, col in frags:
        fill = PAPER if rng.random() < 0.6 else col
        out.append(f'    <polygon points="{scrap(rng, sx, sy, rng.uniform(5, 9))}" fill="{fill}" stroke="{INK}" stroke-width="0.9"/>')
    out.append("  </g>")
    # loose hand-drawn ink fragments
    out.append(f'  <g fill="none" stroke="{INK}" stroke-width="1" stroke-linecap="round" filter="url(#ink)" opacity="0.75">')
    for k in range(7):
        x, y = rng.uniform(180, 340), rng.uniform(120, 380)
        pts = [(x, y)] + [(x + rng.uniform(-14, 14) + 6 * n, y + rng.uniform(-8, 8)) for n in range(1, 4)]
        out.append(f'    <path d="{catmull(pts)}"/>')
    out.append("  </g>")
    svg("retrieval.svg", "Paper scraps and ink fragments drifting toward clustered islands on a quiet coordinate grid", "\n".join(out), 53)


# ---------------------------------------------------------------------------
# 6. Vision / hidden structure vs surface  (Computer Vision)
# ---------------------------------------------------------------------------
def vision():
    rng = random.Random(67)
    out = []
    # surface: field of loose graphite marks on a faint grid
    fx0, fy0, cols, rows, s = 160, 110, 16, 8, 30
    fx, fy, fw, fh = 6, 2, 5, 4  # frame position in cells
    out.append(f'  <g stroke="{PENCIL}" stroke-width="0.9" stroke-linecap="round" filter="url(#ink)">')
    for i in range(cols):
        for j in range(rows):
            if fx <= i < fx + fw and fy <= j < fy + fh:
                continue
            cx, cy = fx0 + i * s + s / 2, fy0 + j * s + s / 2
            a = rng.uniform(0, math.pi)
            L = rng.uniform(4, 10)
            op = rng.uniform(0.35, 0.8)
            out.append(f'    <line x1="{f(cx - L * math.cos(a))}" y1="{f(cy - L * math.sin(a))}" x2="{f(cx + L * math.cos(a))}" y2="{f(cy + L * math.sin(a))}" opacity="{op:.2f}"/>')
    out.append("  </g>")

    # inside the frame: the hidden structure, as ordered tiles
    out.append('  <g filter="url(#torn)">')
    for i in range(fx, fx + fw):
        for j in range(fy, fy + fh):
            x, y = fx0 + i * s + 4, fy0 + j * s + 4
            on = (i - fx, j - fy) in {(1, 1), (2, 1), (2, 2), (3, 2), (1, 2), (2, 3)}
            col = CLAY if on else (SAND if (i + j) % 2 else PAPER)
            out.append(f'    <rect x="{x}" y="{y}" width="{s - 8}" height="{s - 8}" fill="{col}" stroke="{INK}" stroke-width="0.9"/>')
    out.append("  </g>")

    # hand-cut viewfinder corners
    x0, y0 = fx0 + fx * s - 6, fy0 + fy * s - 6
    x1, y1 = fx0 + (fx + fw) * s + 6, fy0 + (fy + fh) * s + 6
    L = 22
    out.append(f'  <g fill="none" stroke="{INK}" stroke-width="2.4" stroke-linecap="square" filter="url(#ink)">')
    out.append(f'    <path d="M{x0},{y0 + L} L{x0},{y0} L{x0 + L},{y0}"/>')
    out.append(f'    <path d="M{x1 - L},{y0} L{x1},{y0} L{x1},{y0 + L}"/>')
    out.append(f'    <path d="M{x1},{y1 - L} L{x1},{y1} L{x1 - L},{y1}"/>')
    out.append(f'    <path d="M{x0 + L},{y1} L{x0},{y1} L{x0},{y1 - L}"/>')
    out.append("  </g>")
    svg("vision.svg", "A field of loose graphite marks where a hand-cut frame reveals an ordered structure of tiles", "\n".join(out), 67)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    compression()
    interpretability()
    evaluation()
    steering()
    retrieval()
    vision()

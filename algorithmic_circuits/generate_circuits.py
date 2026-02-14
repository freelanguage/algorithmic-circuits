#!/usr/bin/env python3
"""
================================================================================
ALGORITHMIC CIRCUITS - Art Generator
================================================================================

Generates 10 cyberpunk circuit board artworks using mathematical algorithms.

Created: January 2026
Author: Agent Zero with Kimi K2.5
License: MIT (Free to use, modify, and distribute)

================================================================================
PLATFORM-SPECIFIC SETUP INSTRUCTIONS
================================================================================

--- LINUX (Ubuntu/Debian/Kali) ---
# 1. Install Python (if not installed)
sudo apt update
sudo apt install python3 python3-pip

# 2. Install required libraries
pip3 install numpy pillow scipy

# 3. Run the generator
cd algorithmic_circuits
python3 generate_circuits.py

# 4. Output appears in "images/" folder

--- LINUX (Fedora/RHEL/CentOS) ---
# 1. Install Python
sudo dnf install python3 python3-pip

# 2. Install libraries
pip3 install numpy pillow scipy

# 3. Run
cd algorithmic_circuits
python3 generate_circuits.py

--- macOS ---
# 1. Install Homebrew (if not installed)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# 2. Install Python
brew install python

# 3. Install libraries
pip3 install numpy pillow scipy

# 4. Run
cd algorithmic_circuits
python3 generate_circuits.py

--- WINDOWS ---
# 1. Install Python from python.org
#    Download: https://www.python.org/downloads/
#    Check "Add Python to PATH" during installation

# 2. Open Command Prompt or PowerShell
#    Press Win+R, type "cmd", press Enter

# 3. Install libraries
python -m pip install numpy pillow scipy

#    OR if you have multiple Python versions:
py -3 -m pip install numpy pillow scipy

# 4. Navigate to folder and run
cd algorithmic_circuits
python generate_circuits.py

#    OR:
py -3 generate_circuits.py

--- TROUBLESHOOTING ---

Issue: "ModuleNotFoundError: No module named numpy"
Fix:   pip3 install numpy pillow scipy  (Linux/macOS)
       python -m pip install numpy pillow scipy  (Windows)

Issue: "Permission denied" on Linux/macOS
Fix:   Add --user flag: pip3 install --user numpy pillow scipy

Issue: Images appear black or corrupted
Fix:   Ensure scipy is properly installed: pip install scipy --upgrade

--- OUTPUT ---
All images save to: ./images/
Format: PNG, 1920x1080 pixels

================================================================================
"""

import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import random
import math
from scipy.spatial import Voronoi, Delaunay
from scipy.ndimage import gaussian_filter

# ============================================================================
# CONFIGURATION
# ============================================================================
WIDTH, HEIGHT = 1920, 1080
OUTPUT_DIR = 'images'

# Neon cyberpunk palette
COLORS = {
    'cyan': (0, 255, 255),
    'magenta': (255, 0, 255),
    'yellow': (255, 255, 0),
    'lime': (0, 255, 128),
    'hot_pink': (255, 0, 128),
    'electric_blue': (0, 128, 255),
    'orange': (255, 128, 0),
    'purple': (128, 0, 255),
    'white': (255, 255, 255),
    'dark_bg': (5, 5, 15),
    'grid': (20, 20, 40)
}

# ============================================================================
# PERLIN NOISE IMPLEMENTATION
# ============================================================================
def fade(t):
    """Fade curve for smooth interpolation"""
    return t * t * t * (t * (t * 6 - 15) + 10)

def grad(h, x, y):
    """Gradient function for hash value"""
    h = h & 3
    if h == 0: return x + y
    elif h == 1: return -x + y
    elif h == 2: return x - y
    else: return -x - y

# Permutation table (classic Perlin)
_perm = [151,160,137,91,90,15,131,13,201,95,96,53,194,233,7,225,140,36,103,30,69,142,8,99,37,240,21,10,23,190,6,148,247,120,234,75,0,26,197,62,94,252,219,203,117,35,11,32,57,177,33,88,237,149,56,87,174,20,125,136,171,168,68,175,74,165,71,134,139,48,27,166,77,146,158,231,83,111,229,122,60,211,133,230,220,105,92,41,55,46,245,40,244,102,143,54,65,25,63,161,1,216,80,73,209,76,132,187,208,89,18,169,200,196,135,130,116,188,159,86,164,100,109,198,173,186,3,64,52,217,226,250,124,123,5,202,38,147,118,126,255,82,85,212,207,206,59,227,47,16,58,17,182,189,28,42,223,183,170,213,119,248,152,2,44,154,163,70,221,153,101,155,167,43,172,9,129,22,39,253,19,98,108,110,79,113,224,232,178,185,112,104,218,246,97,228,251,34,242,193,238,210,144,12,191,179,162,241,81,51,145,235,249,14,239,107,49,192,214,31,181,199,106,157,184,84,204,176,115,121,50,45,127,4,150,254,138,236,205,93,222,114,67,29,24,72,243,141,128,195,78,66,215,61,156,180]
PERM = _perm + _perm

def perlin2d(x, y):
    """2D Perlin noise value at coordinates x, y"""
    X = int(x) & 255
    Y = int(y) & 255
    xf = x - int(x)
    yf = y - int(y)
    u = fade(xf)
    v = fade(yf)
    A = PERM[X] + Y
    B = PERM[X + 1] + Y
    g1 = grad(PERM[A], xf, yf)
    g2 = grad(PERM[B], xf - 1, yf)
    g3 = grad(PERM[A + 1], xf, yf - 1)
    g4 = grad(PERM[B + 1], xf - 1, yf - 1)
    return (g1 + u*(g2-g1)) + v*((g3+u*(g4-g3))-(g1+u*(g2-g1)))

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================
def create_base():
    """Create dark background with subtle grid pattern"""
    img = Image.new('RGB', (WIDTH, HEIGHT), COLORS['dark_bg'])
    draw = ImageDraw.Draw(img)
    for x in range(0, WIDTH, 40):
        draw.line([(x, 0), (x, HEIGHT)], fill=COLORS['grid'], width=1)
    for y in range(0, HEIGHT, 40):
        draw.line([(0, y), (WIDTH, y)], fill=COLORS['grid'], width=1)
    return img

def add_glow(img, color, intensity=0.3):
    """Apply neon glow effect using Gaussian blur compositing"""
    glow = img.filter(ImageFilter.GaussianBlur(radius=15))
    glow_data = np.array(glow)
    mask = (glow_data[:,:,0] > 10) | (glow_data[:,:,1] > 10) | (glow_data[:,:,2] > 10)
    result = np.array(img).copy()
    result[mask] = (result[mask] * (1 - intensity) + glow_data[mask] * intensity).astype(np.uint8)
    return Image.fromarray(result)

def draw_glowing_line(draw, start, end, color, width=2):
    """Draw circuit trace with neon glow halo"""
    r, g, b = color
    for i in range(8, 0, -2):
        alpha = int(50 - i * 5)
        glow_color = (max(0, r-alpha), max(0, g-alpha), max(0, b-alpha))
        draw.line([start, end], fill=glow_color, width=width + i)
    draw.line([start, end], fill=color, width=width)

def draw_glowing_circle(draw, center, radius, color):
    """Draw circuit node with concentric glow rings"""
    r, g, b = color
    for i in range(20, 0, -3):
        alpha = int(60 - i * 2)
        glow_color = (max(0, r-alpha), max(0, g-alpha), max(0, b-alpha))
        draw.ellipse([center[0]-radius-i, center[1]-radius-i, 
                     center[0]+radius+i, center[1]+radius+i], outline=glow_color, width=2)
    draw.ellipse([center[0]-radius, center[1]-radius, 
                 center[0]+radius, center[1]+radius], outline=color, width=3)
    draw.ellipse([center[0]-radius//3, center[1]-radius//3, 
                 center[0]+radius//3, center[1]+radius//3], fill=color)

# ============================================================================
# ARTWORK GENERATORS
# ============================================================================

def generate_01_voronoi():
    """01 - Voronoi Synthesis: Distance-based cellular partitions"""
    img = create_base()
    draw = ImageDraw.Draw(img)
    np.random.seed(42)
    points = np.random.rand(80, 2) * [WIDTH, HEIGHT]
    points = np.vstack([points, [[0,0], [WIDTH,0], [0,HEIGHT], [WIDTH,HEIGHT]]])
    vor = Voronoi(points)
    for ridge in vor.ridge_vertices:
        if -1 not in ridge:
            p1 = tuple(vor.vertices[ridge[0]].astype(int))
            p2 = tuple(vor.vertices[ridge[1]].astype(int))
            if 0 <= p1[0] < WIDTH and 0 <= p1[1] < HEIGHT and 0 <= p2[0] < WIDTH and 0 <= p2[1] < HEIGHT:
                color = random.choice([COLORS['cyan'], COLORS['magenta'], COLORS['electric_blue']])
                if random.random() > 0.3:
                    draw_glowing_line(draw, p1, p2, color, width=2)
    for point in vor.vertices:
        x, y = int(point[0]), int(point[1])
        if 0 <= x < WIDTH and 0 <= y < HEIGHT and random.random() > 0.7:
            draw_glowing_circle(draw, (x, y), random.randint(3, 8), 
                              random.choice([COLORS['yellow'], COLORS['lime']]))
    return add_glow(img, COLORS['cyan'])

def generate_02_lsystem():
    """02 - L-System Traces: String rewriting fractals"""
    img = create_base()
    draw = ImageDraw.Draw(img)
    def l_system():
        axiom, rules = "F+F+F+F", {"F": "F+F-F-FF+F+F-F"}
        result = axiom
        for _ in range(4):
            result = "".join(rules.get(c, c) for c in result)
        return result
    def draw_lsys(instructions, start, angle, length, color):
        x, y, stack = start[0], start[1], []
        for cmd in instructions:
            if cmd == "F":
                rad = math.radians(angle)
                nx, ny = x + length * math.cos(rad), y + length * math.sin(rad)
                draw_glowing_line(draw, (int(x), int(y)), (int(nx), int(ny)), color, width=2)
                x, y = nx, ny
            elif cmd == "+": angle += 90
            elif cmd == "-": angle -= 90
            elif cmd == "[": stack.append((x, y, angle))
            elif cmd == "]": x, y, angle = stack.pop()
    instructions = l_system()
    colors = [COLORS['cyan'], COLORS['magenta'], COLORS['yellow'], COLORS['lime']]
    for i in range(4):
        ox, oy = WIDTH // 4 + (i % 2) * (WIDTH // 2), HEIGHT // 4 + (i // 2) * (HEIGHT // 2)
        draw_lsys(instructions, (ox, oy), i * 90, 8, colors[i])
    draw_glowing_circle(draw, (WIDTH//2, HEIGHT//2), 50, COLORS['hot_pink'])
    return add_glow(img, COLORS['magenta'])

def generate_03_perlin():
    """03 - Perlin Flow: Gradient noise flow fields"""
    img = create_base()
    draw = ImageDraw.Draw(img)
    scale = 0.003
    for y in range(0, HEIGHT, 30):
        for x in range(0, WIDTH, 30):
            n = perlin2d(x * scale, y * scale)
            angle, length = n * 2 * math.pi, 40 + abs(n) * 60
            ex, ey = x + length * math.cos(angle), y + length * math.sin(angle)
            color = COLORS['cyan'] if n > 0.3 else COLORS['electric_blue'] if n > 0 else COLORS['magenta'] if n > -0.3 else COLORS['purple']
            draw_glowing_line(draw, (x, y), (int(ex), int(ey)), color, width=2)
            if abs(n) > 0.5: draw_glowing_circle(draw, (x, y), 5, COLORS['yellow'])
    return add_glow(img, COLORS['electric_blue'])

def generate_04_cellular():
    """04 - Cellular Matrix: Conway's Game of Life"""
    img = create_base()
    pixels = np.array(img)
    grid = np.random.choice([0, 1], size=(HEIGHT//10, WIDTH//10), p=[0.85, 0.15])
    for _ in range(15):
        new_grid = grid.copy()
        for y in range(1, grid.shape[0]-1):
            for x in range(1, grid.shape[1]-1):
                neighbors = np.sum(grid[y-1:y+2, x-1:x+2]) - grid[y, x]
                if grid[y, x] == 1 and (neighbors < 2 or neighbors > 3): new_grid[y, x] = 0
                elif grid[y, x] == 0 and neighbors == 3: new_grid[y, x] = 1
        grid = new_grid
    for y in range(grid.shape[0]):
        for x in range(grid.shape[1]):
            if grid[y, x]:
                px, py = x * 10, y * 10
                color = [COLORS['cyan'], COLORS['magenta'], COLORS['yellow'], COLORS['lime']][(x + y) % 4]
                for dy in range(10):
                    for dx in range(10):
                        if px + dx < WIDTH and py + dy < HEIGHT: pixels[py + dy, px + dx] = color
    img = Image.fromarray(pixels)
    draw = ImageDraw.Draw(img)
    for y in range(0, HEIGHT, 100): draw_glowing_line(draw, (0, y), (WIDTH, y), COLORS['electric_blue'], width=1)
    for x in range(0, WIDTH, 100): draw_glowing_line(draw, (x, 0), (x, HEIGHT), COLORS['electric_blue'], width=1)
    return add_glow(img, COLORS['lime'])

def generate_05_mandelbrot():
    """05 - Mandelbrot Core: Complex number iteration"""
    img = create_base()
    pixels = np.array(img)
    def mandel_iter(c, max_iter=30):
        z = 0
        for n in range(max_iter):
            if abs(z) > 2: return n
            z = z*z + c
        return max_iter
    for px in range(0, WIDTH, 4):
        for py in range(0, HEIGHT, 4):
            x = (px - WIDTH/2) / (WIDTH * 0.8 / 4) - 0.5
            y = (py - HEIGHT/2) / (HEIGHT * 0.8 / 4)
            m = mandel_iter(complex(x, y))
            if m < 30:
                color = [COLORS['cyan'], COLORS['electric_blue'], COLORS['magenta'], COLORS['hot_pink'], COLORS['yellow']][m % 5]
                for dy in range(4):
                    for dx in range(4):
                        if px + dx < WIDTH and py + dy < HEIGHT: pixels[py + dy, px + dx] = color
    img = Image.fromarray(pixels)
    draw = ImageDraw.Draw(img)
    for i in range(20):
        angle = math.radians(i * 18)
        start = (WIDTH//2, HEIGHT//2)
        for r in range(50, min(WIDTH, HEIGHT)//2, 30):
            ex, ey = WIDTH//2 + int(r * math.cos(angle)), HEIGHT//2 + int(r * math.sin(angle))
            if 0 <= ex < WIDTH and 0 <= ey < HEIGHT:
                draw_glowing_line(draw, start, (ex, ey), random.choice([COLORS['cyan'], COLORS['lime'], COLORS['yellow']]), width=2)
                draw_glowing_circle(draw, (ex, ey), 6, COLORS['white'])
            start = (ex, ey)
    return add_glow(img, COLORS['cyan'])

def generate_06_delaunay():
    """06 - Delaunay Network: Optimal mesh triangulation"""
    img = create_base()
    draw = ImageDraw.Draw(img)
    np.random.seed(123)
    points = np.random.rand(120, 2) * [WIDTH, HEIGHT]
    for x in range(100, WIDTH, 200):
        for y in range(100, HEIGHT, 200): points = np.vstack([points, [[x, y]]])
    tri = Delaunay(points)
    for simplex in tri.simplices:
        pts = [tuple(points[i].astype(int)) for i in simplex]
        color = random.choice([COLORS['cyan'], COLORS['magenta'], COLORS['electric_blue'], COLORS['lime']])
        for i in range(3): draw_glowing_line(draw, pts[i], pts[(i+1)%3], color, width=2)
        draw.polygon(pts, fill=(color[0]//10, color[1]//10, color[2]//10))
    for pt in points:
        x, y = int(pt[0]), int(pt[1])
        if 0 <= x < WIDTH and 0 <= y < HEIGHT:
            draw_glowing_circle(draw, (x, y), random.randint(4, 10), 
                              COLORS['yellow'] if random.random() > 0.5 else COLORS['hot_pink'])
    return add_glow(img, COLORS['magenta'])

def generate_07_reaction_diffusion():
    """07 - Reaction Diffusion: Gray-Scott chemical patterns"""
    img = create_base()
    pixels = np.array(img).astype(float)
    u, v = np.ones((HEIGHT, WIDTH)), np.zeros((HEIGHT, WIDTH))
    cx, cy = WIDTH // 2, HEIGHT // 2
    u[cy-50:cy+50, cx-50:cx+50] = 0.5
    v[cy-50:cy+50, cx-50:cx+50] = 0.25
    Du, Dv, f, k, dt = 0.16, 0.08, 0.035, 0.06, 1.0
    for _ in range(50):
        Lu, Lv = gaussian_filter(u, sigma=1) - u, gaussian_filter(v, sigma=1) - v
        uv2 = u * v * v
        u += (Du * Lu - uv2 + f * (1 - u)) * dt
        v += (Dv * Lv + uv2 - (f + k) * v) * dt
    v_norm = (v - v.min()) / (v.max() - v.min())
    for y in range(HEIGHT):
        for x in range(WIDTH):
            val = v_norm[y, x]
            if val > 0.3:
                pixels[y, x] = COLORS['cyan'] if val > 0.7 else COLORS['electric_blue'] if val > 0.55 else COLORS['magenta'] if val > 0.4 else COLORS['purple']
    img = Image.fromarray(pixels.astype(np.uint8))
    draw = ImageDraw.Draw(img)
    for i in range(0, WIDTH, 80): draw_glowing_line(draw, (i, 0), (i, HEIGHT), COLORS['lime'], width=1)
    for i in range(0, HEIGHT, 80): draw_glowing_line(draw, (0, i), (WIDTH, i), COLORS['lime'], width=1)
    return add_glow(img, COLORS['cyan'])

def generate_08_koch():
    """08 - Koch Crystals: Recursive geometric fractals"""
    img = create_base()
    draw = ImageDraw.Draw(img)
    def koch_line(p1, p2, depth, color):
        if depth == 0:
            draw_glowing_line(draw, p1, p2, color, width=2)
            return
        x1, y1, x2, y2 = p1[0], p1[1], p2[0], p2[1]
        dx, dy = (x2 - x1) / 3, (y2 - y1) / 3
        a, b = (x1, y1), (x1 + dx, y1 + dy)
        d, e = (x1 + 2*dx, y1 + 2*dy), (x2, y2)
        angle = math.atan2(dy, dx) - math.pi/3
        side = math.sqrt(dx*dx + dy*dy)
        c = (b[0] + side * math.cos(angle), b[1] + side * math.sin(angle))
        koch_line(a, b, depth-1, color); koch_line(b, c, depth-1, color)
        koch_line(c, d, depth-1, color); koch_line(d, e, depth-1, color)
    colors = [COLORS['cyan'], COLORS['magenta'], COLORS['yellow'], COLORS['lime'], COLORS['electric_blue']]
    for i, color in enumerate(colors):
        size, cx, cy = 150 + i * 50, WIDTH//2 + (i - 2) * 200, HEIGHT//2
        for angle in range(0, 360, 60):
            rad, rad2 = math.radians(angle), math.radians(angle + 60)
            p1, p2 = (cx + size * math.cos(rad), cy + size * math.sin(rad)), (cx + size * math.cos(rad2), cy + size * math.sin(rad2))
            koch_line(p1, p2, 3, color)
    for i in range(5):
        y = 150 + i * 180
        draw_glowing_line(draw, (0, y), (WIDTH, y), COLORS['hot_pink'], width=1)
    return add_glow(img, COLORS['yellow'])

def generate_09_flowfield():
    """09 - Flowfield Traces: Particle vector fields"""
    img = create_base()
    draw = ImageDraw.Draw(img)
    scale, num_particles, steps = 0.005, 200, 150
    for p in range(num_particles):
        x, y = random.randint(0, WIDTH), random.randint(0, HEIGHT)
        points = [(x, y)]
        for _ in range(steps):
            angle = perlin2d(x * scale, y * scale) * 4 * math.pi
            x += 3 * math.cos(angle); y += 3 * math.sin(angle)
            points.append((int(x), int(y)))
            if not (0 <= x < WIDTH and 0 <= y < HEIGHT): break
        color = random.choice([COLORS['cyan'], COLORS['magenta'], COLORS['electric_blue'], COLORS['lime']])
        for i in range(len(points) - 1):
            if random.random() > 0.7: draw_glowing_line(draw, points[i], points[i+1], color, width=2)
    for _ in range(50):
        x, y = random.randint(0, WIDTH), random.randint(0, HEIGHT)
        draw_glowing_circle(draw, (x, y), random.randint(5, 12), COLORS['yellow'])
    return add_glow(img, COLORS['electric_blue'])

def generate_10_sierpinski():
    """10 - Sierpinski Mesh: Recursive topological voids"""
    img = create_base()
    draw = ImageDraw.Draw(img)
    def draw_triangle(p1, p2, p3, depth, max_depth):
        if depth == 0:
            color = [COLORS['cyan'], COLORS['magenta'], COLORS['yellow'], COLORS['lime']][max_depth % 4]
            draw_glowing_line(draw, p1, p2, color, width=2)
            draw_glowing_line(draw, p2, p3, color, width=2)
            draw_glowing_line(draw, p3, p1, color, width=2)
            draw_glowing_circle(draw, ((p1[0]+p2[0]+p3[0])//3, (p1[1]+p2[1]+p3[1])//3), 4, COLORS['white'])
            return
        m12 = ((p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2)
        m23 = ((p2[0] + p3[0]) // 2, (p2[1] + p3[1]) // 2)
        m31 = ((p3[0] + p1[0]) // 2, (p3[1] + p1[1]) // 2)
        draw_triangle(p1, m12, m31, depth-1, max_depth)
        draw_triangle(m12, p2, m23, depth-1, max_depth)
        draw_triangle(m31, m23, p3, depth-1, max_depth)
    centers = [(WIDTH//4, HEIGHT//3), (3*WIDTH//4, HEIGHT//3), (WIDTH//2, 2*HEIGHT//3)]
    for center, size, depth in zip(centers, [300, 250, 280], [5, 4, 5]):
        cx, cy = center
        p1, p2, p3 = (cx, cy - size), (cx - size * 0.866, cy + size * 0.5), (cx + size * 0.866, cy + size * 0.5)
        draw_triangle(p1, p2, p3, depth, depth)
    for i, (c1, c2) in enumerate(zip(centers, centers[1:] + [centers[0]])):
        draw_glowing_line(draw, c1, c2, [COLORS['cyan'], COLORS['magenta'], COLORS['yellow']][i], width=3)
    return add_glow(img, COLORS['yellow'])

# ============================================================================
# MAIN EXECUTION
# ============================================================================
if __name__ == "__main__":
    import os
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    generators = [
        ("01_voronoi_synthesis", generate_01_voronoi),
        ("02_lsystem_traces", generate_02_lsystem),
        ("03_perlin_flow", generate_03_perlin),
        ("04_cellular_matrix", generate_04_cellular),
        ("05_mandelbrot_core", generate_05_mandelbrot),
        ("06_delaunay_network", generate_06_delaunay),
        ("07_reaction_diffusion", generate_07_reaction_diffusion),
        ("08_koch_crystals", generate_08_koch),
        ("09_flowfield_traces", generate_09_flowfield),
        ("10_sierpinski_mesh", generate_10_sierpinski),
    ]

    print("=" * 60)
    print("ALGORITHMIC CIRCUITS - Art Generator")
    print("=" * 60)
    print(f"\nGenerating {len(generators)} artworks...\n")

    for name, gen in generators:
        print(f"  Creating {name}...", end=" ")
        try:
            img = gen()
            img.save(f'{OUTPUT_DIR}/{name}.png', 'PNG')
            print("✓")
        except Exception as e:
            print(f"✗ {e}")

    print("\n" + "=" * 60)
    print("COMPLETE! Artworks saved to:", OUTPUT_DIR)
    print("=" * 60)

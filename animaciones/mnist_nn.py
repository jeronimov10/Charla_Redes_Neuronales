import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.animation import FuncAnimation

np.random.seed(42)

# ── Paleta ────────────────────────────────────────────────────────────────
BG            = "#000000"
TEXT_COLOR    = "white"
SUBTEXT_COLOR = "#cbd5e1"
COLOR_EDGE    = "#2563eb"
COLOR_ACTIVE  = "#22c55e"
COLOR_NODE    = "#1f1f1f"
COLOR_BORDER  = "#d9d9d9"

# ── Mock imagen 5×5 del dígito "3" ───────────────────────────────────────
PIXEL_GRID = np.array([
    [1, 1, 1, 1, 0],
    [0, 0, 0, 1, 0],
    [0, 1, 1, 1, 0],
    [0, 0, 0, 1, 0],
    [1, 1, 1, 1, 0],
], dtype=float)
ROWS, COLS = PIXEL_GRID.shape

# ── Activaciones simuladas ────────────────────────────────────────────────
act_input  = np.array([0.9, 0.0, 1.0, 0.85, 0.0, 0.95])
act_hidden = np.array([0.78, 0.42, 0.91, 0.35])
act_output = np.array([0.07, 0.05, 0.08, 0.88])
OUTPUT_LABELS = ['"0"', '"1"', '"2"', '"3"']
WINNER = 3

# ── Posiciones ────────────────────────────────────────────────────────────
GRID_CX, GRID_CY = 1.4, 4.6
CELL = 0.42
GX0 = GRID_CX - (COLS * CELL) / 2
GY0 = GRID_CY - (ROWS * CELL) / 2

LX = [4.0, 7.0, 10.0]
LAYER_N = [6, 4, 4]

def layer_ys(n, cy=4.6, spacing=1.15):
    return [cy + (n - 1) / 2 * spacing - i * spacing for i in range(n)]

positions = [
    [(LX[li], y) for y in layer_ys(n)]
    for li, n in enumerate(LAYER_N)
]

# ── Figura ────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(15, 8.5))
fig.patch.set_facecolor(BG)
ax.set_facecolor(BG)
ax.set_xlim(0, 12.5)
ax.set_ylim(0, 10.2)
ax.axis("off")

# IMPORTANTE: reservar espacio arriba y evitar tight_layout
fig.subplots_adjust(left=0.04, right=0.98, bottom=0.06, top=0.92)

# ── Títulos (más arriba y con separación real) ───────────────────────────
title_txt = ax.text(
    6.25, 9.55, "",
    color=TEXT_COLOR, fontsize=21,
    ha="center", va="center", fontweight="bold",
    clip_on=False
)

sub_txt = ax.text(
    6.25, 9.05, "",
    color=SUBTEXT_COLOR, fontsize=11.5,
    ha="center", va="center",
    clip_on=False,
    wrap=True
)

# ── Pixel grid ────────────────────────────────────────────────────────────
grid_rects = {}
for r in range(ROWS):
    for c in range(COLS):
        x = GX0 + c * CELL
        y = GY0 + (ROWS - 1 - r) * CELL
        rect = mpatches.Rectangle(
            (x, y), CELL * 0.9, CELL * 0.9,
            facecolor="#111111", edgecolor="#2a2a2a", lw=0.8, zorder=3
        )
        ax.add_patch(rect)
        grid_rects[(r, c)] = rect

ax.text(
    GRID_CX, GY0 + ROWS * CELL + 0.22, "Imagen de entrada",
    color="white", fontsize=11, ha="center", fontweight="bold"
)
ax.text(
    GRID_CX, GY0 - 0.38, "(5×5 pixels)",
    color=SUBTEXT_COLOR, fontsize=9, ha="center"
)

# ── Flecha flatten ────────────────────────────────────────────────────────
flatten_arr = ax.annotate(
    "", xy=(LX[0] - 0.38, GRID_CY), xytext=(GX0 + COLS * CELL + 0.08, GRID_CY),
    arrowprops=dict(arrowstyle="->", color=COLOR_EDGE, lw=2.0), zorder=2
)
flatten_lbl = ax.text(
    (GX0 + COLS * CELL + LX[0]) / 2, GRID_CY + 0.45,
    "flatten", color=SUBTEXT_COLOR, fontsize=10, ha="center"
)

# ── Nombres de capas (más abajo para que no choquen con subtítulo) ──────
layer_name_y = 7.45
for li, name in enumerate(["Input (6/25)", "Hidden", "Output"]):
    ax.text(
        LX[li], layer_name_y, name,
        color="white", fontsize=13,
        ha="center", fontweight="bold"
    )

# ── Etiquetas de salida ───────────────────────────────────────────────────
for i, lbl in enumerate(OUTPUT_LABELS):
    x, y = positions[2][i]
    ax.text(x + 0.72, y, lbl, color="white", fontsize=13, va="center")

# ── Nota de puntos suspensivos en capa input ──────────────────────────────
xs, ys_last = positions[0][-1]
ax.text(xs, ys_last - 0.55, "⋮", color=SUBTEXT_COLOR, fontsize=16, ha="center")

# ── Aristas ───────────────────────────────────────────────────────────────
edge_lines = {}
for li in range(len(LAYER_N) - 1):
    for i, (x1, y1) in enumerate(positions[li]):
        for j, (x2, y2) in enumerate(positions[li + 1]):
            ln, = ax.plot(
                [x1, x2], [y1, y2],
                color=COLOR_EDGE, lw=1.0, alpha=0.35, zorder=1
            )
            edge_lines[(li, i, j)] = ln

# ── Nodos ─────────────────────────────────────────────────────────────────
node_circles = {}
node_texts = {}
for li, layer in enumerate(positions):
    for ni, (x, y) in enumerate(layer):
        c = plt.Circle(
            (x, y), 0.28,
            facecolor=COLOR_NODE,
            edgecolor=COLOR_BORDER, lw=1.5, zorder=3
        )
        ax.add_patch(c)
        node_circles[(li, ni)] = c

        t = ax.text(
            x, y, "",
            color="white", fontsize=9,
            ha="center", va="center", zorder=4, fontweight="bold"
        )
        node_texts[(li, ni)] = t

# ── Texto resultado final ─────────────────────────────────────────────────
result_txt = ax.text(
    11.5, GRID_CY, "",
    color=COLOR_ACTIVE, fontsize=20,
    ha="center", va="center", fontweight="bold"
)

# ── Helpers ───────────────────────────────────────────────────────────────
def rgba_hex(hex_color, alpha):
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i+2], 16) / 255 for i in (0, 2, 4))
    return (r, g, b, max(0.0, min(1.0, alpha)))

def node_color(v, progress=1.0):
    alpha = 0.2 + 0.8 * min(abs(v) * progress, 1.0)
    return rgba_hex(COLOR_ACTIVE, alpha)

def show_grid(progress=1.0):
    for (r, c), rect in grid_rects.items():
        v = PIXEL_GRID[r, c]
        if v > 0:
            rect.set_facecolor(rgba_hex(COLOR_ACTIVE, 0.15 + 0.75 * progress))
        else:
            rect.set_facecolor("#111111")

def activate_layer(li, acts, progress=1.0):
    for ni, v in enumerate(acts):
        node_circles[(li, ni)].set_facecolor(node_color(v, progress))
        if progress >= 0.55:
            node_texts[(li, ni)].set_text(f"{v * progress:.2f}")

def activate_edges(li, progress=1.0):
    for i in range(LAYER_N[li]):
        for j in range(LAYER_N[li + 1]):
            ln = edge_lines[(li, i, j)]
            ln.set_color(COLOR_ACTIVE)
            ln.set_linewidth(1.0 + 1.8 * progress)
            ln.set_alpha(0.3 + 0.6 * progress)

def reset():
    title_txt.set_text("")
    sub_txt.set_text("")
    result_txt.set_text("")

    for rect in grid_rects.values():
        rect.set_facecolor("#111111")

    for c in node_circles.values():
        c.set_facecolor(COLOR_NODE)
        c.set_edgecolor(COLOR_BORDER)
        c.set_linewidth(1.5)

    for t in node_texts.values():
        t.set_text("")

    for ln in edge_lines.values():
        ln.set_color(COLOR_EDGE)
        ln.set_linewidth(1.0)
        ln.set_alpha(0.35)

def p(frame, start, end):
    return min(1.0, max(0.0, (frame - start) / (end - start)))

# ── Animación ─────────────────────────────────────────────────────────────
FRAMES = 210

def update(frame):
    reset()

    if frame < 22:
        title_txt.set_text("Reconocimiento de Dígitos Escritos a Mano")
        sub_txt.set_text("Red neuronal — forward propagation paso a paso")

    elif frame < 55:
        title_txt.set_text("1. Imagen de Entrada")
        sub_txt.set_text("Cada pixel = 0 (blanco) o 1 (negro) → 25 valores numéricos")
        show_grid(progress=p(frame, 22, 45))

    elif frame < 95:
        title_txt.set_text("2. Flatten — Aplanar la Imagen")
        sub_txt.set_text("Los 25 pixels se convierten en un vector 1D → entradas de la red")
        show_grid()
        activate_layer(0, act_input, progress=p(frame, 55, 88))

    elif frame < 140:
        title_txt.set_text("3. Forward Propagation → Capa Oculta")
        sub_txt.set_text("Cada neurona detecta un patrón distinto: bordes, curvas, ángulos...")
        show_grid()
        activate_layer(0, act_input)
        activate_edges(0, progress=p(frame, 95, 118))
        activate_layer(1, act_hidden, progress=p(frame, 110, 135))

    elif frame < 175:
        title_txt.set_text("4. Forward Propagation → Capa de Salida")
        sub_txt.set_text("Cada neurona de salida = probabilidad de ser ese dígito")
        show_grid()
        activate_layer(0, act_input)
        activate_layer(1, act_hidden)
        activate_edges(0)
        activate_edges(1, progress=p(frame, 140, 163))
        activate_layer(2, act_output, progress=p(frame, 155, 172))

    else:
        title_txt.set_text("5. Predicción Final")
        sub_txt.set_text(
            f'Neurona más activa → dígito {OUTPUT_LABELS[WINNER]} '
            f'({act_output[WINNER] * 100:.0f}% confianza)'
        )
        show_grid()
        activate_layer(0, act_input)
        activate_layer(1, act_hidden)
        activate_edges(0)
        activate_edges(1)
        activate_layer(2, act_output)

        node_circles[(2, WINNER)].set_edgecolor("#ffffff")
        node_circles[(2, WINNER)].set_linewidth(3.5)
        pulse = 0.85 + 0.15 * np.sin((frame - 175) * 0.5)
        node_circles[(2, WINNER)].set_facecolor(rgba_hex(COLOR_ACTIVE, pulse))
        result_txt.set_text(f'→ {OUTPUT_LABELS[WINNER]}')

    artists = [title_txt, sub_txt, result_txt, flatten_lbl]
    artists += list(grid_rects.values())
    artists += list(edge_lines.values())
    artists += list(node_circles.values())
    artists += list(node_texts.values())
    return artists

ani = FuncAnimation(fig, update, frames=FRAMES, interval=105, blit=False, repeat=True)

# Para guardar como GIF:
ani.save("mnist_nn_animation.gif", writer="pillow", fps=10)

plt.show()

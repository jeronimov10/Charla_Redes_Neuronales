import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

np.random.seed(42)

layer_sizes = [3, 4, 1]

COLOR_NODE = "#1f1f1f"
COLOR_NODE_BORDER = "#d9d9d9"
COLOR_FORWARD = "#22c55e"
COLOR_OUTPUT = "#16a34a"
COLOR_EDGE = "#2563eb"
COLOR_EDGE_ACTIVE = "#22c55e"
BG_COLOR = "#000000"
TEXT_COLOR = "white"
SUBTEXT_COLOR = "#cbd5e1"

input_data = np.array([25.0, 70.0, 60.0])

activations = [
    input_data / np.array([40, 100, 100]),
    np.array([0.3, 0.7, 0.4, 0.6]),
    np.array([0.75])
]

input_labels = ["Temperatura", "Humedad", "Nubosidad"]
output_labels = ["Prob. Lluvia"]

def get_node_positions(layer_sizes):
    positions = []
    x_positions = [0.0, 3.8, 7.6]
    for i, size in enumerate(layer_sizes):
        x = x_positions[i]
        y_positions = np.linspace(-(size - 1), (size - 1), size) * 0.9
        positions.append([(x, y) for y in y_positions])
    return positions

node_positions = get_node_positions(layer_sizes)

edges = []
for l in range(len(layer_sizes) - 1):
    for i in range(layer_sizes[l]):
        for j in range(layer_sizes[l + 1]):
            edges.append(((l, i), (l + 1, j)))

fig, ax = plt.subplots(figsize=(13, 7))
fig.patch.set_facecolor(BG_COLOR)
ax.set_facecolor(BG_COLOR)
ax.set_xlim(-2.2, 9.6)
ax.set_ylim(-3.6, 4.8)
ax.axis("off")

title_text = ax.text(
    3.7, 4.2, "",
    color=TEXT_COLOR, fontsize=20, ha="center", va="center", fontweight="bold"
)

subtitle_text = ax.text(
    3.7, 3.72, "",
    color=SUBTEXT_COLOR, fontsize=12, ha="center", va="center"
)

layer_names = ["Input", "Hidden", "Output"]
layer_name_y = 3.15 

for l, name in enumerate(layer_names):
    x = node_positions[l][0][0]
    ax.text(x, layer_name_y, name, color="white", fontsize=14, ha="center", va="center", fontweight="bold")

for i, label in enumerate(input_labels):
    x, y = node_positions[0][i]
    ax.text(x - 1.2, y, label, color="white", fontsize=10, ha="right", va="center")

x, y = node_positions[-1][0]
ax.text(x + 1.25, y, output_labels[0], color="white", fontsize=10, ha="left", va="center")

edge_lines = {}
for edge in edges:
    (l1, i1), (l2, i2) = edge
    x1, y1 = node_positions[l1][i1]
    x2, y2 = node_positions[l2][i2]
    line, = ax.plot(
        [x1, x2], [y1, y2],
        color=COLOR_EDGE, lw=1.2, alpha=0.4, zorder=1
    )
    edge_lines[edge] = line

node_circles = {}
node_texts = {}
for l, layer in enumerate(node_positions):
    for i, (x, y) in enumerate(layer):
        circle = plt.Circle(
            (x, y), 0.28, color=COLOR_NODE, ec=COLOR_NODE_BORDER, lw=1.6, zorder=3
        )
        ax.add_patch(circle)
        node_circles[(l, i)] = circle

        txt = ax.text(
            x, y, "",
            color="white", fontsize=9, ha="center", va="center",
            zorder=4, fontweight="bold"
        )
        node_texts[(l, i)] = txt

def reset_visuals():
    for line in edge_lines.values():
        line.set_color(COLOR_EDGE)
        line.set_linewidth(1.2)
        line.set_alpha(0.4)

    for circle in node_circles.values():
        circle.set_facecolor(COLOR_NODE)
        circle.set_edgecolor(COLOR_NODE_BORDER)
        circle.set_linewidth(1.6)

    for txt in node_texts.values():
        txt.set_text("")
        txt.set_color("white")

    title_text.set_text("")
    subtitle_text.set_text("")

def rgba_from_hex(hex_color, alpha):
    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[i:i+2], 16) / 255 for i in (0, 2, 4)) + (alpha,)

def color_from_value(v, mode="forward"):
    v = abs(v)
    alpha = 0.3 + 0.7 * min(v, 1.0)
    if mode == "forward":
        return rgba_from_hex(COLOR_FORWARD, alpha)
    return rgba_from_hex(COLOR_OUTPUT, alpha)

def show_layer_values(layer_idx, values, mode="forward"):
    for i, v in enumerate(values):
        circle = node_circles[(layer_idx, i)]
        txt = node_texts[(layer_idx, i)]
        if layer_idx == len(layer_sizes) - 1:
            circle.set_facecolor(color_from_value(v, "output"))
        else:
            circle.set_facecolor(color_from_value(v, "forward"))
        txt.set_text(f"{v:.2f}")
        txt.set_color("white")

def activate_connections(layer_idx):
    for i in range(layer_sizes[layer_idx]):
        for j in range(layer_sizes[layer_idx + 1]):
            edge = ((layer_idx, i), (layer_idx + 1, j))
            line = edge_lines[edge]
            line.set_color(COLOR_EDGE_ACTIVE)
            line.set_linewidth(2.5)
            line.set_alpha(0.9)

frames_total = 150

def update(frame):
    reset_visuals()

    if frame < 20:
        title_text.set_text("Predicción de lluvia con red neuronal")
        subtitle_text.set_text("Sistema de IA para pronóstico meteorológico")

    elif frame < 50:
        title_text.set_text("1. Datos de entrada")
        subtitle_text.set_text("Temperatura, humedad y nubosidad")
        show_layer_values(0, activations[0], mode="forward")

    elif frame < 80:
        title_text.set_text("2. Procesamiento en capa oculta")
        subtitle_text.set_text("La red combina patrones climáticos")
        show_layer_values(0, activations[0], mode="forward")
        activate_connections(0)
        progress = min(1.0, (frame - 50) / 25)
        show_layer_values(1, activations[1] * progress, mode="forward")

    elif frame < 110:
        title_text.set_text("3. Cálculo de probabilidad")
        subtitle_text.set_text("La red genera la predicción final")
        show_layer_values(0, activations[0], mode="forward")
        show_layer_values(1, activations[1], mode="forward")
        activate_connections(0)
        activate_connections(1)
        progress = min(1.0, (frame - 80) / 25)
        show_layer_values(2, activations[2] * progress, mode="forward")

    else:
        title_text.set_text("Predicción final")
        prob = activations[2][0]
        if prob >= 0.5:
            subtitle_text.set_text(f"Probabilidad de lluvia: {prob*100:.1f}%  |  VA A LLOVER")
        else:
            subtitle_text.set_text(f"Probabilidad de lluvia: {prob*100:.1f}%  |  NO VA A LLOVER")

        show_layer_values(0, activations[0], mode="forward")
        show_layer_values(1, activations[1], mode="forward")
        show_layer_values(2, activations[2], mode="forward")
        activate_connections(0)
        activate_connections(1)

        circle = node_circles[(2, 0)]
        circle.set_edgecolor("#ffffff")
        circle.set_linewidth(2.6)

    artists = [title_text, subtitle_text]
    artists += list(edge_lines.values())
    artists += list(node_circles.values())
    artists += list(node_texts.values())
    return artists

ani = FuncAnimation(
    fig, update, frames=frames_total, interval=100, blit=False, repeat=True
)

plt.tight_layout()

print("Guardando animacion...")
ani.save("animacion_clima.gif", writer="pillow", fps=12)
print("GIF guardado: animacion_clima.gif")

plt.show()
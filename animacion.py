import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

np.random.seed(42)

layer_sizes = [4, 5, 4, 2]

COLOR_NODE = "#1f1f1f"
COLOR_NODE_BORDER = "#d9d9d9"
COLOR_FORWARD = "#22c55e"
COLOR_OUTPUT = "#16a34a"
COLOR_BACKPROP = "#86efac"
COLOR_EDGE = "#2563eb"
COLOR_EDGE_ACTIVE = "#22c55e"
COLOR_EDGE_BACK = "#4ade80"
BG_COLOR = "#000000"
TEXT_COLOR = "white"
SUBTEXT_COLOR = "#cbd5e1"

activations = [
    np.array([0.2, 0.9, 0.4, 0.7]),
    np.array([0.1, 0.8, 0.5, 0.6, 0.2]),
    np.array([0.9, 0.3, 0.7, 0.4]),
    np.array([0.15, 0.85])
]

errors = [
    np.array([0.0, 0.0, 0.0, 0.0]),
    np.array([0.15, -0.1, 0.2, -0.05, 0.1]),
    np.array([0.25, -0.15, 0.2, -0.1]),
    np.array([-0.7, 0.7])
]

output_labels = ["Clase A", "Clase B"]
target = np.array([0, 1])

def get_node_positions(layer_sizes):
    positions = []
    x_spacing = 3.0
    for i, size in enumerate(layer_sizes):
        x = i * x_spacing
        y_positions = np.linspace(-(size - 1), (size - 1), size) * 0.95
        positions.append([(x, y) for y in y_positions])
    return positions

node_positions = get_node_positions(layer_sizes)

edges = []
for l in range(len(layer_sizes) - 1):
    for i in range(layer_sizes[l]):
        for j in range(layer_sizes[l + 1]):
            edges.append(((l, i), (l + 1, j)))

fig, ax = plt.subplots(figsize=(15, 8.5))
fig.patch.set_facecolor(BG_COLOR)
ax.set_facecolor(BG_COLOR)
ax.set_xlim(-1.5, 10.5)
ax.set_ylim(-5.2, 6.8)
ax.axis("off")

title_text = ax.text(
    4.5, 6.0, "",
    color=TEXT_COLOR, fontsize=24, ha="center", va="center", fontweight="bold"
)

subtitle_text = ax.text(
    4.5, 5.3, "",
    color=SUBTEXT_COLOR, fontsize=13, ha="center", va="center"
)

loss_text = ax.text(
    8.9, -4.6, "",
    color=COLOR_FORWARD, fontsize=14, ha="center", va="center", fontweight="bold"
)

layer_names = ["Input", "Hidden 1", "Hidden 2", "Output"]
for l, name in enumerate(layer_names):
    x = node_positions[l][0][0]
    ax.text(x, 4.85, name, color="white", fontsize=16, ha="center", fontweight="bold")

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
            (x, y), 0.28, color=COLOR_NODE, ec=COLOR_NODE_BORDER, lw=1.8, zorder=3
        )
        ax.add_patch(circle)
        node_circles[(l, i)] = circle

        txt = ax.text(
            x, y, "",
            color="white", fontsize=10, ha="center", va="center",
            zorder=4, fontweight="bold"
        )
        node_texts[(l, i)] = txt

for i, label in enumerate(output_labels):
    x, y = node_positions[-1][i]
    ax.text(x + 0.9, y, label, color="white", fontsize=12, va="center")

def reset_visuals():
    for line in edge_lines.values():
        line.set_color(COLOR_EDGE)
        line.set_linewidth(1.2)
        line.set_alpha(0.4)

    for circle in node_circles.values():
        circle.set_facecolor(COLOR_NODE)
        circle.set_edgecolor(COLOR_NODE_BORDER)
        circle.set_linewidth(1.8)

    for txt in node_texts.values():
        txt.set_text("")
        txt.set_color("white")

    title_text.set_text("")
    subtitle_text.set_text("")
    loss_text.set_text("")

def rgba_from_hex(hex_color, alpha):
    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[i:i+2], 16)/255 for i in (0, 2, 4)) + (alpha,)

def color_from_value(v, mode="forward"):
    v = abs(v)
    alpha = 0.2 + 0.8 * min(v, 1.0)
    if mode == "forward":
        return rgba_from_hex(COLOR_FORWARD, alpha)
    if mode == "output":
        return rgba_from_hex(COLOR_OUTPUT, alpha)
    return rgba_from_hex(COLOR_BACKPROP, alpha)

def show_layer_values(layer_idx, values, mode="forward"):
    for i, v in enumerate(values):
        circle = node_circles[(layer_idx, i)]
        txt = node_texts[(layer_idx, i)]
        if mode == "forward":
            if layer_idx == len(layer_sizes) - 1:
                circle.set_facecolor(color_from_value(v, "output"))
            else:
                circle.set_facecolor(color_from_value(v, "forward"))
        else:
            circle.set_facecolor(color_from_value(v, "backprop"))
        txt.set_text(f"{v:.2f}")
        txt.set_color("white")

def activate_connections(layer_idx, mode="forward"):
    for i in range(layer_sizes[layer_idx]):
        for j in range(layer_sizes[layer_idx + 1]):
            edge = ((layer_idx, i), (layer_idx + 1, j))
            line = edge_lines[edge]
            if mode == "forward":
                line.set_color(COLOR_EDGE_ACTIVE)
                line.set_linewidth(2.6)
                line.set_alpha(0.95)
            else:
                line.set_color(COLOR_EDGE_BACK)
                line.set_linewidth(2.6)
                line.set_alpha(0.95)

def highlight_output():
    predicted_idx = int(np.argmax(activations[-1]))
    for i in range(layer_sizes[-1]):
        circle = node_circles[(len(layer_sizes) - 1, i)]
        if i == predicted_idx:
            circle.set_edgecolor("#ffffff")
            circle.set_linewidth(3.0)

def show_error():
    pred = activations[-1]
    loss = np.mean((pred - target) ** 2)
    title_text.set_text("4. Error Calculation")
    subtitle_text.set_text("La red compara su predicción con la respuesta correcta")
    loss_text.set_text(f"Loss = {loss:.4f}")
    for i, err in enumerate(errors[-1]):
        circle = node_circles[(len(layer_sizes) - 1, i)]
        circle.set_facecolor(color_from_value(err, "backprop"))
        node_texts[(len(layer_sizes) - 1, i)].set_text(f"{err:+.2f}")

frames_total = 220

def update(frame):
    reset_visuals()

    if frame < 20:
        title_text.set_text("How a Neural Network Learns")
        subtitle_text.set_text("Forward propagation → Prediction → Error → Backpropagation → Weight update")

    elif frame < 50:
        title_text.set_text("1. Input Layer")
        subtitle_text.set_text("La red recibe los datos de entrada")
        show_layer_values(0, activations[0], mode="forward")

    elif frame < 80:
        title_text.set_text("2. Forward Propagation")
        subtitle_text.set_text("La información avanza hacia la primera capa oculta")
        show_layer_values(0, activations[0], mode="forward")
        activate_connections(0, mode="forward")
        progress = min(1.0, (frame - 50) / 20)
        show_layer_values(1, activations[1] * progress, mode="forward")

    elif frame < 110:
        title_text.set_text("2. Forward Propagation")
        subtitle_text.set_text("La red combina patrones en capas internas")
        show_layer_values(0, activations[0], mode="forward")
        show_layer_values(1, activations[1], mode="forward")
        activate_connections(0, mode="forward")
        activate_connections(1, mode="forward")
        progress = min(1.0, (frame - 80) / 20)
        show_layer_values(2, activations[2] * progress, mode="forward")

    elif frame < 140:
        title_text.set_text("3. Output Layer")
        subtitle_text.set_text("La red produce una predicción final")
        show_layer_values(0, activations[0], mode="forward")
        show_layer_values(1, activations[1], mode="forward")
        show_layer_values(2, activations[2], mode="forward")
        activate_connections(0, mode="forward")
        activate_connections(1, mode="forward")
        activate_connections(2, mode="forward")
        progress = min(1.0, (frame - 110) / 20)
        show_layer_values(3, activations[3] * progress, mode="forward")
        if frame > 128:
            highlight_output()

    elif frame < 160:
        show_layer_values(0, activations[0], mode="forward")
        show_layer_values(1, activations[1], mode="forward")
        show_layer_values(2, activations[2], mode="forward")
        show_layer_values(3, activations[3], mode="forward")
        activate_connections(0, mode="forward")
        activate_connections(1, mode="forward")
        activate_connections(2, mode="forward")
        highlight_output()
        show_error()

    elif frame < 190:
        title_text.set_text("5. Backpropagation")
        subtitle_text.set_text("El error se propaga hacia atrás para ajustar los pesos")
        show_layer_values(3, errors[3], mode="backprop")
        activate_connections(2, mode="backprop")
        progress = min(1.0, (frame - 160) / 18)
        show_layer_values(2, errors[2] * progress, mode="backprop")

    elif frame < 210:
        title_text.set_text("5. Backpropagation")
        subtitle_text.set_text("Las capas internas reciben la señal del error")
        show_layer_values(3, errors[3], mode="backprop")
        show_layer_values(2, errors[2], mode="backprop")
        activate_connections(2, mode="backprop")
        activate_connections(1, mode="backprop")
        progress = min(1.0, (frame - 190) / 14)
        show_layer_values(1, errors[1] * progress, mode="backprop")

    else:
        title_text.set_text("6. Learning / Weight Update")
        subtitle_text.set_text("La red ajusta sus conexiones para mejorar")
        show_layer_values(1, errors[1], mode="backprop")
        show_layer_values(2, errors[2], mode="backprop")
        show_layer_values(3, errors[3], mode="backprop")
        activate_connections(0, mode="backprop")
        activate_connections(1, mode="backprop")
        activate_connections(2, mode="backprop")
        loss_text.set_text("Weights updated")
        pulse = 1.5 + 1.0 * np.sin((frame - 210) * 0.7)
        for line in edge_lines.values():
            line.set_linewidth(max(1.2, pulse))
            line.set_alpha(0.95)

    artists = [title_text, subtitle_text, loss_text]
    artists += list(edge_lines.values())
    artists += list(node_circles.values())
    artists += list(node_texts.values())
    return artists

ani = FuncAnimation(
    fig, update, frames=frames_total, interval=120, blit=False, repeat=True
)

plt.tight_layout()

# # Guardar como GIF
# print("Guardando como GIF...")
# ani.save("neural_network_animation.gif", writer="pillow", fps=10)
# print("GIF guardado: neural_network_animation.gif")

plt.show()
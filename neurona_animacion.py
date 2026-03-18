import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Circle

x = np.array([0.8, 0.4, 0.9])
w = np.array([0.7, -0.5, 0.9])
b = 0.2

weighted = x * w
z = weighted.sum() + b
a = max(0, z)

fig, ax = plt.subplots(figsize=(10, 6))
fig.patch.set_facecolor("black")
ax.set_facecolor("black")
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis("off")

input_pos = [(1.5, 7.5), (1.5, 5.0), (1.5, 2.5)]
neuron_pos = (6.2, 5.0)
output_pos = (8.7, 5.0)

for x0, y0 in input_pos:
    ax.plot([x0 + 0.45, neuron_pos[0] - 0.6], [y0, neuron_pos[1]], color="#2563eb", lw=2, alpha=0.35)

ax.plot([neuron_pos[0] + 0.6, output_pos[0] - 0.45], [neuron_pos[1], output_pos[1]], color="#2563eb", lw=2, alpha=0.35)

input_circles = []
input_texts = []
weight_texts = []

for i, (px, py) in enumerate(input_pos):
    c = Circle((px, py), 0.42, facecolor="#111111", edgecolor="white", lw=2)
    ax.add_patch(c)
    input_circles.append(c)

    t = ax.text(px, py, f"x{i+1}\n{x[i]:.1f}", color="white", ha="center", va="center", fontsize=11, fontweight="bold")
    input_texts.append(t)

    mx = (px + neuron_pos[0]) / 2 - 0.2
    my = (py + neuron_pos[1]) / 2
    wt = ax.text(mx, my, f"w{i+1}={w[i]:.1f}", color="#93c5fd", ha="center", va="center", fontsize=10)
    weight_texts.append(wt)

neuron_circle = Circle(neuron_pos, 0.6, facecolor="#111111", edgecolor="white", lw=2.5)
ax.add_patch(neuron_circle)

output_circle = Circle(output_pos, 0.42, facecolor="#111111", edgecolor="white", lw=2)
ax.add_patch(output_circle)

title = ax.text(5, 9.35, "Cómo funciona una neurona artificial", color="white", ha="center", va="center", fontsize=20, fontweight="bold")
subtitle = ax.text(5, 8.65, "", color="#cbd5e1", ha="center", va="center", fontsize=12)

neuron_text = ax.text(neuron_pos[0], neuron_pos[1], "Σ", color="white", ha="center", va="center", fontsize=22, fontweight="bold")
formula_text = ax.text(5, 1.0, "", color="#86efac", ha="center", va="center", fontsize=14, fontweight="bold")
output_text = ax.text(output_pos[0], output_pos[1], "", color="white", ha="center", va="center", fontsize=11, fontweight="bold")
bias_text = ax.text(neuron_pos[0], neuron_pos[1] - 1.15, "", color="#fca5a5", ha="center", va="center", fontsize=11)

frames = 120

def lerp_color(c1, c2, t):
    import matplotlib.colors as mcolors
    a = np.array(mcolors.to_rgb(c1))
    b = np.array(mcolors.to_rgb(c2))
    c = (1 - t) * a + t * b
    return c

def update(frame):
    for c in input_circles:
        c.set_facecolor("#111111")
    neuron_circle.set_facecolor("#111111")
    output_circle.set_facecolor("#111111")
    subtitle.set_text("")
    formula_text.set_text("")
    output_text.set_text("")
    bias_text.set_text("")
    neuron_text.set_text("Σ")

    for line in ax.lines:
        line.set_color("#2563eb")
        line.set_alpha(0.35)
        line.set_linewidth(2)

    if frame < 20:
        subtitle.set_text("La neurona recibe varias entradas")
        k = min(1, frame / 15)
        for c in input_circles:
            c.set_facecolor(lerp_color("#111111", "#22c55e", k))

    elif frame < 45:
        subtitle.set_text("Cada entrada se multiplica por su peso")
        p = frame - 20
        active_idx = min(2, p // 8)
        for i in range(active_idx + 1):
            input_circles[i].set_facecolor("#22c55e")
            ax.lines[i].set_color("#22c55e")
            ax.lines[i].set_alpha(0.95)
            ax.lines[i].set_linewidth(3)
        shown = min(active_idx + 1, 3)
        parts = [f"({x[i]:.1f}×{w[i]:.1f})" for i in range(shown)]
        formula_text.set_text(" + ".join(parts))

    elif frame < 65:
        subtitle.set_text("Se suman los valores ponderados y el bias")
        for i in range(3):
            input_circles[i].set_facecolor("#22c55e")
            ax.lines[i].set_color("#22c55e")
            ax.lines[i].set_alpha(0.95)
            ax.lines[i].set_linewidth(3)
        k = min(1, (frame - 45) / 15)
        neuron_circle.set_facecolor(lerp_color("#111111", "#22c55e", k))
        formula_text.set_text(f"z = {weighted[0]:.2f} + {weighted[1]:.2f} + {weighted[2]:.2f} + {b:.2f} = {z:.2f}")
        bias_text.set_text(f"bias = {b:.2f}")

    elif frame < 90:
        subtitle.set_text("La función de activación transforma el resultado")
        for i in range(3):
            input_circles[i].set_facecolor("#22c55e")
            ax.lines[i].set_color("#22c55e")
            ax.lines[i].set_alpha(0.95)
            ax.lines[i].set_linewidth(3)
        neuron_circle.set_facecolor("#22c55e")
        neuron_text.set_text("ReLU")
        formula_text.set_text(f"a = max(0, z) = max(0, {z:.2f}) = {a:.2f}")
        bias_text.set_text(f"z = {z:.2f}")

    else:
        subtitle.set_text("La neurona produce una salida")
        for i in range(3):
            input_circles[i].set_facecolor("#22c55e")
            ax.lines[i].set_color("#22c55e")
            ax.lines[i].set_alpha(0.95)
            ax.lines[i].set_linewidth(3)
        neuron_circle.set_facecolor("#22c55e")
        neuron_text.set_text("ReLU")
        ax.lines[3].set_color("#22c55e")
        ax.lines[3].set_alpha(0.95)
        ax.lines[3].set_linewidth(3)
        k = min(1, (frame - 90) / 15)
        output_circle.set_facecolor(lerp_color("#111111", "#22c55e", k))
        output_text.set_text(f"salida\n{a:.2f}")
        formula_text.set_text("Entrada → suma ponderada → activación → salida")

ani = FuncAnimation(fig, update, frames=frames, interval=90, repeat=True)

ani.save("neurona_animada.gif", writer=PillowWriter(fps=12))

plt.show()
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

np.random.seed(1)

BG = "#000000"
TEXT = "white"
SUBTEXT = "#cbd5e1"
BLUE = "#2563eb"
GREEN = "#22c55e"
GREEN_DARK = "#16a34a"
ORANGE = "#f59e0b"
RED = "#ef4444"
GRAY = "#1f1f1f"
BORDER = "#d9d9d9"

fig, ax = plt.subplots(figsize=(14, 8))
fig.patch.set_facecolor(BG)
ax.set_facecolor(BG)
ax.set_xlim(0, 14)
ax.set_ylim(0, 10)
ax.axis("off")

title = ax.text(7, 9.4, "", color=TEXT, fontsize=22, ha="center", va="center", fontweight="bold")
subtitle = ax.text(7, 8.8, "", color=SUBTEXT, fontsize=12, ha="center", va="center")

input_pos = (1.7, 5)
hidden_pos = [(4.5, 6.7), (4.5, 5), (4.5, 3.3)]
output_pos = (7.5, 5)
target_box_pos = (10.5, 6.2)
loss_box_pos = (10.5, 3.8)
update_box_pos = (12.3, 5)

input_circle = plt.Circle(input_pos, 0.5, facecolor=GRAY, edgecolor=BORDER, lw=2, zorder=3)
ax.add_patch(input_circle)

hidden_circles = []
for pos in hidden_pos:
    c = plt.Circle(pos, 0.45, facecolor=GRAY, edgecolor=BORDER, lw=2, zorder=3)
    ax.add_patch(c)
    hidden_circles.append(c)

output_circle = plt.Circle(output_pos, 0.5, facecolor=GRAY, edgecolor=BORDER, lw=2, zorder=3)
ax.add_patch(output_circle)

edge_lines = []
for hp in hidden_pos:
    line, = ax.plot([input_pos[0] + 0.5, hp[0] - 0.45], [input_pos[1], hp[1]], color=BLUE, lw=2, alpha=0.35)
    edge_lines.append(line)

for hp in hidden_pos:
    line, = ax.plot([hp[0] + 0.45, output_pos[0] - 0.5], [hp[1], output_pos[1]], color=BLUE, lw=2, alpha=0.35)
    edge_lines.append(line)

ax.text(input_pos[0], 6.0, "Input", color=TEXT, fontsize=14, ha="center", fontweight="bold")
ax.text(4.5, 7.7, "Hidden Layer", color=TEXT, fontsize=14, ha="center", fontweight="bold")
ax.text(output_pos[0], 6.0, "Output", color=TEXT, fontsize=14, ha="center", fontweight="bold")

input_text = ax.text(input_pos[0], input_pos[1], "x", color=TEXT, fontsize=14, ha="center", va="center", fontweight="bold")
hidden_texts = [
    ax.text(p[0], p[1], "", color=TEXT, fontsize=11, ha="center", va="center", fontweight="bold")
    for p in hidden_pos
]
output_text = ax.text(output_pos[0], output_pos[1], "", color=TEXT, fontsize=13, ha="center", va="center", fontweight="bold")

target_box = plt.Rectangle((9.4, 5.6), 2.2, 0.9, facecolor=GRAY, edgecolor=BORDER, lw=2)
loss_box = plt.Rectangle((9.4, 3.2), 2.2, 0.9, facecolor=GRAY, edgecolor=BORDER, lw=2)
update_box = plt.Rectangle((11.5, 4.55), 1.6, 0.9, facecolor=GRAY, edgecolor=BORDER, lw=2)

ax.add_patch(target_box)
ax.add_patch(loss_box)
ax.add_patch(update_box)

target_title = ax.text(10.5, 6.75, "Valor real", color=TEXT, fontsize=13, ha="center", fontweight="bold")
loss_title = ax.text(10.5, 4.35, "Error / Loss", color=TEXT, fontsize=13, ha="center", fontweight="bold")
update_title = ax.text(12.3, 5.8, "Ajuste", color=TEXT, fontsize=13, ha="center", fontweight="bold")

target_text = ax.text(10.5, 6.05, "", color=TEXT, fontsize=13, ha="center", va="center", fontweight="bold")
loss_text = ax.text(10.5, 3.65, "", color=TEXT, fontsize=13, ha="center", va="center", fontweight="bold")
update_text = ax.text(12.3, 5.0, "", color=TEXT, fontsize=12, ha="center", va="center", fontweight="bold")

formula_text = ax.text(7, 1.1, "", color=GREEN, fontsize=14, ha="center", va="center", fontweight="bold")

prediction = 0.78
target = 1.00
loss = (target - prediction) ** 2
grad = -0.44
lr = 0.1

particles = []
for _ in range(6):
    p, = ax.plot([], [], "o", color=GREEN, markersize=6, alpha=0)
    particles.append(p)

back_particles = []
for _ in range(6):
    p, = ax.plot([], [], "o", color=ORANGE, markersize=6, alpha=0)
    back_particles.append(p)

def reset_scene():
    input_circle.set_facecolor(GRAY)
    output_circle.set_facecolor(GRAY)
    output_circle.set_edgecolor(BORDER)
    output_circle.set_linewidth(2)

    for c in hidden_circles:
        c.set_facecolor(GRAY)
        c.set_edgecolor(BORDER)
        c.set_linewidth(2)

    for line in edge_lines:
        line.set_color(BLUE)
        line.set_alpha(0.35)
        line.set_linewidth(2)

    input_text.set_text("x")
    output_text.set_text("")
    for t in hidden_texts:
        t.set_text("")

    target_text.set_text("")
    loss_text.set_text("")
    update_text.set_text("")
    formula_text.set_text("")
    title.set_text("")
    subtitle.set_text("")

    target_box.set_facecolor(GRAY)
    loss_box.set_facecolor(GRAY)
    update_box.set_facecolor(GRAY)

    for p in particles + back_particles:
        p.set_alpha(0)

def move_particle(p, start, end, t):
    x = start[0] + (end[0] - start[0]) * t
    y = start[1] + (end[1] - start[1]) * t
    p.set_data([x], [y])
    p.set_alpha(1)

frames_total = 220

def update(frame):
    reset_scene()

    if frame < 20:
        title.set_text("Aprendizaje de una red neuronal")
        subtitle.set_text("Predicción → comparación → error → backpropagation → descenso de gradiente")

    elif frame < 50:
        title.set_text("1. Forward propagation")
        subtitle.set_text("La red procesa la entrada y genera una predicción")
        input_circle.set_facecolor(GREEN)
        input_text.set_text("0.65")

        for i, c in enumerate(hidden_circles):
            c.set_facecolor(GREEN)
            hidden_texts[i].set_text(["0.32", "0.71", "0.44"][i])

        for line in edge_lines:
            line.set_color(GREEN)
            line.set_alpha(0.9)
            line.set_linewidth(2.7)

        progress = min(1.0, (frame - 20) / 25)
        output_circle.set_facecolor((0.13, 0.77, 0.37, 0.2 + 0.8 * progress))
        output_text.set_text(f"{prediction:.2f}")

        for i in range(3):
            move_particle(particles[i], input_pos, hidden_pos[i], progress)
        for i in range(3):
            move_particle(particles[i + 3], hidden_pos[i], output_pos, progress)

    elif frame < 85:
        title.set_text("2. Comparación con el valor real")
        subtitle.set_text("La predicción se compara con la respuesta correcta")
        input_circle.set_facecolor(GREEN)
        output_circle.set_facecolor(GREEN_DARK)
        output_circle.set_edgecolor("white")
        output_circle.set_linewidth(2.5)
        input_text.set_text("0.65")
        output_text.set_text(f"{prediction:.2f}")

        for i, c in enumerate(hidden_circles):
            c.set_facecolor(GREEN)
            hidden_texts[i].set_text(["0.32", "0.71", "0.44"][i])

        for line in edge_lines:
            line.set_color(GREEN)
            line.set_alpha(0.85)
            line.set_linewidth(2.6)

        target_box.set_facecolor((0.13, 0.77, 0.37, 0.2))
        target_text.set_text(f"{target:.2f}")
        formula_text.set_text("La red predijo 0.78, pero el valor correcto era 1.00")

    elif frame < 125:
        title.set_text("3. Cálculo del error")
        subtitle.set_text("Se mide qué tan lejos estuvo la predicción del valor real")
        input_circle.set_facecolor(GREEN)
        output_circle.set_facecolor(GREEN_DARK)
        input_text.set_text("0.65")
        output_text.set_text(f"{prediction:.2f}")

        for i, c in enumerate(hidden_circles):
            c.set_facecolor(GREEN)
            hidden_texts[i].set_text(["0.32", "0.71", "0.44"][i])

        target_box.set_facecolor((0.13, 0.77, 0.37, 0.2))
        target_text.set_text(f"{target:.2f}")

        loss_box.set_facecolor((0.94, 0.27, 0.27, 0.22))
        loss_text.set_text(f"{loss:.4f}")
        formula_text.set_text("Loss = (y - ŷ)² = (1.00 - 0.78)²")

    elif frame < 170:
        title.set_text("4. Backpropagation")
        subtitle.set_text("El error se propaga hacia atrás para saber qué pesos deben cambiar")
        input_circle.set_facecolor(GRAY)
        input_text.set_text("0.65")
        output_circle.set_facecolor(ORANGE)
        output_text.set_text(f"{prediction:.2f}")
        output_circle.set_edgecolor("white")
        output_circle.set_linewidth(2.5)

        for i, c in enumerate(hidden_circles):
            c.set_facecolor((0.96, 0.62, 0.04, 0.35))
            hidden_texts[i].set_text(["∂1", "∂2", "∂3"][i])

        target_box.set_facecolor((0.13, 0.77, 0.37, 0.2))
        target_text.set_text(f"{target:.2f}")
        loss_box.set_facecolor((0.94, 0.27, 0.27, 0.22))
        loss_text.set_text(f"{loss:.4f}")

        for i, line in enumerate(edge_lines):
            if i >= 3:
                line.set_color(ORANGE)
            else:
                line.set_color("#60a5fa")
            line.set_alpha(0.9)
            line.set_linewidth(2.7)

        progress = min(1.0, (frame - 125) / 35)
        for i in range(3):
            move_particle(back_particles[i], output_pos, hidden_pos[i], progress)
        for i in range(3):
            move_particle(back_particles[i + 3], hidden_pos[i], input_pos, progress)

        formula_text.set_text("Backpropagation calcula cómo contribuyó cada peso al error")

    else:
        title.set_text("5. Descenso de gradiente")
        subtitle.set_text("Los pesos se actualizan para reducir el error en la siguiente iteración")

        update_box.set_facecolor((0.13, 0.77, 0.37, 0.25))
        update_text.set_text("w = w - η·∇L")

        loss_box.set_facecolor((0.94, 0.27, 0.27, 0.22))
        loss_text.set_text(f"grad = {grad:.2f}")

        target_box.set_facecolor((0.13, 0.77, 0.37, 0.2))
        target_text.set_text(f"lr = {lr:.2f}")

        for i, c in enumerate(hidden_circles):
            pulse = 0.3 + 0.7 * abs(np.sin((frame - 170) * 0.25))
            c.set_facecolor((0.13, 0.77, 0.37, pulse))
            hidden_texts[i].set_text("Δw")

        output_circle.set_facecolor(GREEN_DARK)
        output_text.set_text("mejora")
        output_circle.set_edgecolor("white")
        output_circle.set_linewidth(2.5)

        for line in edge_lines:
            pulse = 2 + 1.2 * abs(np.sin((frame - 170) * 0.25))
            line.set_color(GREEN)
            line.set_alpha(0.95)
            line.set_linewidth(pulse)

        formula_text.set_text("Gradient descent mueve los pesos en la dirección que reduce la pérdida")

    artists = [
        title, subtitle, input_text, output_text, target_text, loss_text,
        update_text, formula_text, target_title, loss_title, update_title
    ]
    artists += edge_lines
    artists += [input_circle, output_circle, target_box, loss_box, update_box]
    artists += hidden_circles
    artists += hidden_texts
    artists += particles
    artists += back_particles
    return artists

ani = FuncAnimation(fig, update, frames=frames_total, interval=95, blit=False, repeat=True)

plt.tight_layout()

print("Guardando animación...")
ani.save("aprendizaje_red_neuronal.gif", writer="pillow", fps=12)
print("GIF guardado: aprendizaje_red_neuronal.gif")

plt.show()
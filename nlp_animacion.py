import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle, Rectangle
from textblob import TextBlob

# Oración corta de prueba en vez de archivo
text = "I love this product, it is absolutely amazing and wonderful"

blob = TextBlob(text)
polarity = blob.sentiment.polarity

if polarity > 0.15:
    label = "Positive"
    label_color = "#22c55e"
elif polarity < -0.15:
    label = "Negative"
    label_color = "#ef4444"
else:
    label = "Neutral"
    label_color = "#facc15"

words = text.split()
tokens = words[:8] if len(words) >= 8 else words + ["..."] * (8 - len(words))

np.random.seed(42)
embedding_values = np.clip(np.random.normal(0.55, 0.18, len(tokens)), 0.1, 0.95)
hidden_values = np.array([0.32, 0.61, 0.48, 0.77, 0.43])
output_values = {
    "Positive": np.array([0.84, 0.10, 0.06]),
    "Neutral":  np.array([0.12, 0.76, 0.12]),
    "Negative": np.array([0.08, 0.14, 0.78]),
}[label]

# ── Colores ──
BG      = "#000000"
TEXT    = "white"
SUBTEXT = "#cbd5e1"
BLUE    = "#2563eb"
GREEN   = "#22c55e"
RED     = "#ef4444"
YELLOW  = "#facc15"
GRAY    = "#1f1f1f"
BORDER  = "#d9d9d9"

# ── Figura ──
fig, ax = plt.subplots(figsize=(18, 10))
fig.patch.set_facecolor(BG)
ax.set_facecolor(BG)
ax.set_xlim(0, 18)
ax.set_ylim(0, 11)
ax.axis("off")

# ── Título y subtítulo (zona superior) ──
title    = ax.text(9, 10.4, "", color=TEXT, fontsize=22, ha="center", va="center", fontweight="bold")
subtitle = ax.text(9, 9.85, "", color=SUBTEXT, fontsize=12, ha="center", va="center")

# ══════════════════════════════════════════
# CAPA 1 – Input Text  (x: 0.5 – 4.5)
# ══════════════════════════════════════════
ax.text(2.5, 9.2, "Input Text", color=TEXT, fontsize=14, ha="center", fontweight="bold")

text_box = Rectangle((0.5, 7.2), 4.0, 1.7, facecolor=GRAY, edgecolor=BORDER, lw=2, zorder=2)
ax.add_patch(text_box)

display_text = text if len(text) <= 60 else text[:57] + "..."
text_content = ax.text(
    2.5, 8.05, display_text, color=TEXT, fontsize=9,
    ha="center", va="center", wrap=True,
    bbox=dict(boxstyle="square,pad=0", facecolor="none", edgecolor="none"),
    zorder=3,
)

# ── Tokens (debajo del input) ──
ax.text(2.5, 6.7, "Tokenization", color=TEXT, fontsize=13, ha="center", fontweight="bold")

token_boxes   = []
token_texts   = []
token_centers = []
start_x = 0.55
for i in range(8):
    x = start_x + i * 0.52
    box = Rectangle((x, 5.7), 0.46, 0.6, facecolor=GRAY, edgecolor=BORDER, lw=1.3, zorder=2)
    ax.add_patch(box)
    token_boxes.append(box)
    cx, cy = x + 0.23, 6.0
    token_centers.append((cx, cy))
    txt = ax.text(cx, cy, "", color=TEXT, fontsize=6.5, ha="center", va="center", zorder=3)
    token_texts.append(txt)

# ══════════════════════════════════════════
# CAPA 2 – Embeddings  (x ≈ 6.5)
# ══════════════════════════════════════════
embedding_x = 6.8
ax.text(embedding_x, 9.2, "Embeddings", color=TEXT, fontsize=14, ha="center", fontweight="bold")

embedding_y_values = np.linspace(2.8, 8.6, 8)
embedding_circles  = []
embedding_texts    = []
for y in embedding_y_values:
    c = Circle((embedding_x, y), 0.28, facecolor=GRAY, edgecolor=BORDER, lw=1.5, zorder=2)
    ax.add_patch(c)
    embedding_circles.append(c)
    t = ax.text(embedding_x, y, "", color=TEXT, fontsize=7.5, ha="center", va="center", fontweight="bold", zorder=3)
    embedding_texts.append(t)

# ══════════════════════════════════════════
# CAPA 3 – Hidden / Neural Network (x ≈ 10.5)
# ══════════════════════════════════════════
hidden_x = 10.8
ax.text(hidden_x, 9.2, "Neural Network", color=TEXT, fontsize=14, ha="center", fontweight="bold")

hidden_y_values = np.linspace(3.8, 7.8, 5)
hidden_circles  = []
hidden_texts    = []
for y in hidden_y_values:
    c = Circle((hidden_x, y), 0.35, facecolor=GRAY, edgecolor=BORDER, lw=1.7, zorder=2)
    ax.add_patch(c)
    hidden_circles.append(c)
    t = ax.text(hidden_x, y, "", color=TEXT, fontsize=8.5, ha="center", va="center", fontweight="bold", zorder=3)
    hidden_texts.append(t)

# ══════════════════════════════════════════
# CAPA 4 – Sentiment Output  (x ≈ 14.5)
# ══════════════════════════════════════════
output_x = 14.5
ax.text(output_x, 9.2, "Sentiment Output", color=TEXT, fontsize=14, ha="center", fontweight="bold")

output_labels = ["Positive", "Neutral", "Negative"]
output_colors = [GREEN, YELLOW, RED]
output_y_values = [7.0, 5.8, 4.6]
output_circles = []
output_texts   = []
for y in output_y_values:
    c = Circle((output_x, y), 0.38, facecolor=GRAY, edgecolor=BORDER, lw=1.8, zorder=2)
    ax.add_patch(c)
    output_circles.append(c)
    t = ax.text(output_x, y, "", color=TEXT, fontsize=8.5, ha="center", va="center", fontweight="bold", zorder=3)
    output_texts.append(t)

for i, y in enumerate(output_y_values):
    ax.text(output_x + 0.85, y, output_labels[i], color=TEXT, fontsize=11, va="center")

# ── Conexiones ──
lines_token_embed = []
for i in range(8):
    x1, y1 = token_centers[i]
    x2, y2 = embedding_x, embedding_y_values[i]
    line, = ax.plot([x1 + 0.25, x2 - 0.28], [y1, y2], color=BLUE, lw=1.2, alpha=0.25, zorder=1)
    lines_token_embed.append(line)

lines_embed_hidden = []
for y1 in embedding_y_values:
    for y2 in hidden_y_values:
        line, = ax.plot([embedding_x + 0.28, hidden_x - 0.35], [y1, y2], color=BLUE, lw=0.8, alpha=0.15, zorder=1)
        lines_embed_hidden.append(line)

lines_hidden_output = []
for y1 in hidden_y_values:
    for y2 in output_y_values:
        line, = ax.plot([hidden_x + 0.35, output_x - 0.38], [y1, y2], color=BLUE, lw=1.0, alpha=0.18, zorder=1)
        lines_hidden_output.append(line)

# ── Textos inferiores ──
formula_text = ax.text(9, 1.6, "", color=SUBTEXT, fontsize=12, ha="center", va="center", fontweight="bold")
result_text  = ax.text(9, 0.8, "", color=label_color, fontsize=18, ha="center", va="center", fontweight="bold")

# ── Partículas ──
particles = []
for _ in range(18):
    p, = ax.plot([], [], "o", color=GREEN, markersize=4, alpha=0, zorder=5)
    particles.append(p)


# ── Helpers ──
def move_particle(artist, start, end, t, color=GREEN):
    t = max(0.0, min(1.0, t))
    x = start[0] + (end[0] - start[0]) * t
    y = start[1] + (end[1] - start[1]) * t
    artist.set_data([x], [y])
    artist.set_alpha(1)
    artist.set_color(color)


def rgba(hex_color, alpha):
    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[i : i + 2], 16) / 255 for i in (0, 2, 4)) + (alpha,)


def reset_scene():
    title.set_text("")
    subtitle.set_text("")
    formula_text.set_text("")
    result_text.set_text("")

    for box in token_boxes:
        box.set_facecolor(GRAY)
        box.set_edgecolor(BORDER)
    for txt in token_texts:
        txt.set_text("")

    for c in embedding_circles:
        c.set_facecolor(GRAY)
        c.set_edgecolor(BORDER)
        c.set_linewidth(1.5)
    for t in embedding_texts:
        t.set_text("")

    for c in hidden_circles:
        c.set_facecolor(GRAY)
        c.set_edgecolor(BORDER)
        c.set_linewidth(1.7)
    for t in hidden_texts:
        t.set_text("")

    for c in output_circles:
        c.set_facecolor(GRAY)
        c.set_edgecolor(BORDER)
        c.set_linewidth(1.8)
    for t in output_texts:
        t.set_text("")

    for line in lines_token_embed:
        line.set_color(BLUE)
        line.set_alpha(0.25)
        line.set_linewidth(1.2)
    for line in lines_embed_hidden:
        line.set_color(BLUE)
        line.set_alpha(0.15)
        line.set_linewidth(0.8)
    for line in lines_hidden_output:
        line.set_color(BLUE)
        line.set_alpha(0.18)
        line.set_linewidth(1.0)

    for p in particles:
        p.set_alpha(0)


# ── Animación ──
frames_total = 220


def update(frame):
    reset_scene()

    # ── Fase 0: Intro ──
    if frame < 20:
        title.set_text("NLP Neural Network for Sentiment Analysis")
        subtitle.set_text("Text → Tokens → Embeddings → Neural Processing → Sentiment Prediction")
        formula_text.set_text(f"TextBlob polarity: {polarity:.3f}")

    # ── Fase 1: Tokenización ──
    elif frame < 55:
        title.set_text("1. Tokenization")
        subtitle.set_text("The input text is split into smaller units called tokens")
        n_visible = min(8, max(0, frame - 20) // 4 + 1)
        for i in range(n_visible):
            token_boxes[i].set_facecolor(rgba(GREEN, 0.35))
            token_texts[i].set_text(tokens[i][:7])
        formula_text.set_text("The sentence is broken into words or sub-word tokens")

    # ── Fase 2: Embeddings ──
    elif frame < 95:
        title.set_text("2. Embeddings")
        subtitle.set_text("Each token is converted into a numerical vector")
        for i in range(8):
            token_boxes[i].set_facecolor(rgba(GREEN, 0.25))
            token_texts[i].set_text(tokens[i][:7])

        n = min(8, max(0, frame - 55) // 5 + 1)
        for i in range(n):
            lines_token_embed[i].set_color(GREEN)
            lines_token_embed[i].set_alpha(0.9)
            lines_token_embed[i].set_linewidth(2.0)
            embedding_circles[i].set_facecolor(rgba(GREEN, 0.25 + 0.6 * embedding_values[i]))
            embedding_texts[i].set_text(f"{embedding_values[i]:.2f}")
            t_progress = min(1.0, (frame - 55 - i * 5) / 5)
            move_particle(particles[i], token_centers[i], (embedding_x, embedding_y_values[i]), t_progress)

        formula_text.set_text("Words become vectors so the model can process meaning mathematically")

    # ── Fase 3: Red neuronal ──
    elif frame < 145:
        title.set_text("3. Neural Processing")
        subtitle.set_text("The neural network combines patterns to understand overall sentiment")
        for i in range(8):
            token_boxes[i].set_facecolor(rgba(GREEN, 0.18))
            token_texts[i].set_text(tokens[i][:7])
            embedding_circles[i].set_facecolor(rgba(GREEN, 0.25 + 0.6 * embedding_values[i]))
            embedding_texts[i].set_text(f"{embedding_values[i]:.2f}")
            lines_token_embed[i].set_color(GREEN)
            lines_token_embed[i].set_alpha(0.5)

        for line in lines_embed_hidden:
            line.set_color(GREEN)
            line.set_alpha(0.3)
            line.set_linewidth(1.2)

        progress = min(1.0, (frame - 95) / 30)
        partial = hidden_values * progress
        for i, v in enumerate(partial):
            hidden_circles[i].set_facecolor(rgba(GREEN, 0.25 + 0.65 * min(v, 1.0)))
            hidden_texts[i].set_text(f"{v:.2f}")

        for i in range(5):
            idx = i + 8
            move_particle(
                particles[idx],
                (embedding_x, embedding_y_values[min(i + 1, 7)]),
                (hidden_x, hidden_y_values[i]),
                min(1.0, (frame - 95) / 20),
            )

        formula_text.set_text("Hidden layers detect patterns like tone, intensity and context")

    # ── Fase 4: Predicción ──
    elif frame < 190:
        title.set_text("4. Sentiment Prediction")
        subtitle.set_text("The network outputs probabilities for each sentiment class")

        for i in range(8):
            token_boxes[i].set_facecolor(rgba(GREEN, 0.12))
            token_texts[i].set_text(tokens[i][:7])
            embedding_circles[i].set_facecolor(rgba(GREEN, 0.25 + 0.6 * embedding_values[i]))
            embedding_texts[i].set_text(f"{embedding_values[i]:.2f}")

        for i, v in enumerate(hidden_values):
            hidden_circles[i].set_facecolor(rgba(GREEN, 0.25 + 0.65 * min(v, 1.0)))
            hidden_texts[i].set_text(f"{v:.2f}")

        for line in lines_embed_hidden:
            line.set_color(GREEN)
            line.set_alpha(0.2)
        for line in lines_hidden_output:
            line.set_color(GREEN)
            line.set_alpha(0.45)
            line.set_linewidth(1.6)

        progress = min(1.0, (frame - 145) / 25)
        partial_out = output_values * progress
        for i, v in enumerate(partial_out):
            output_circles[i].set_facecolor(rgba(output_colors[i], 0.25 + 0.65 * min(v, 1.0)))
            output_texts[i].set_text(f"{v:.2f}")

        for i in range(5):
            move_particle(
                particles[13 + (i % 5)],
                (hidden_x, hidden_y_values[i]),
                (output_x, output_y_values[np.argmax(output_values)]),
                min(1.0, (frame - 145) / 18),
                color=label_color,
            )

        formula_text.set_text("The largest output score becomes the final prediction")

    # ── Fase 5: Resultado final ──
    else:
        title.set_text("Final Classification")
        subtitle.set_text("The model selects the sentiment with the highest score")

        for i in range(8):
            token_boxes[i].set_facecolor(rgba(GREEN, 0.1))
            token_texts[i].set_text(tokens[i][:7])
            embedding_circles[i].set_facecolor(rgba(GREEN, 0.25 + 0.6 * embedding_values[i]))
            embedding_texts[i].set_text(f"{embedding_values[i]:.2f}")

        for i, v in enumerate(hidden_values):
            hidden_circles[i].set_facecolor(rgba(GREEN, 0.25 + 0.65 * min(v, 1.0)))
            hidden_texts[i].set_text(f"{v:.2f}")

        for i, v in enumerate(output_values):
            output_circles[i].set_facecolor(rgba(output_colors[i], 0.25 + 0.65 * min(v, 1.0)))
            output_texts[i].set_text(f"{v:.2f}")

        winner = np.argmax(output_values)
        output_circles[winner].set_edgecolor("white")
        output_circles[winner].set_linewidth(3.5)

        for line in lines_hidden_output:
            line.set_color(label_color)
            line.set_alpha(0.3)
            line.set_linewidth(1.4)

        formula_text.set_text(f"Final decision: {label} (highest activation)")
        result_text.set_text(f"Predicted Sentiment: {label}  |  Polarity: {polarity:.3f}")

    return []


ani = FuncAnimation(fig, update, frames=frames_total, interval=95, blit=False, repeat=True)

plt.tight_layout(pad=1.5)

print("Guardando animación...")
ani.save("animacion_red_neuronal_nlp.gif", writer="pillow", fps=12)
print("GIF guardado: animacion_red_neuronal_nlp.gif")

plt.show()
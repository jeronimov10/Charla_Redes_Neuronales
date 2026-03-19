import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle, Rectangle

np.random.seed(42)

# ── Paleta de colores ──
BG         = "#000000"
TEXT       = "white"
SUBTEXT    = "#cbd5e1"
BLUE       = "#2563eb"
GREEN      = "#22c55e"
GREEN_DARK = "#16a34a"
ORANGE     = "#f59e0b"
RED        = "#ef4444"
PURPLE     = "#8b5cf6"
GRAY       = "#1f1f1f"
BORDER     = "#d9d9d9"

# ── Ejemplo: pregunta del usuario ──
user_question  = "¿Que es una neurona artificial?"
tokens         = ["que", "es", "una", "neurona", "artificial"]
vocab_sample   = ["activac.", "artific.", "backprop", "capas", "datos", "es", "funcion", "neurona"]
bow_values     = [0, 1, 0, 0, 0, 1, 0, 1]        # "artific.", "es", "neurona" → 1
intents_short  = ["greeting", "how_nn", "neuron*", "activation", "learning"]
output_probs   = [0.04, 0.08, 0.81, 0.05, 0.02]
predicted_idx  = 2                                  # neuron_structure gana
predicted_tag  = "neuron_structure"
response_text  = "Una neurona artificial\nrecibe entradas, aplica\npesos y usa ReLU\npara activarse."

hidden1_vals   = np.array([0.72, 0.31, 0.88, 0.15, 0.64, 0.43])
hidden2_vals   = np.array([0.55, 0.21, 0.93, 0.67, 0.38])

# ── Figura ──
fig, ax = plt.subplots(figsize=(20, 10))
fig.patch.set_facecolor(BG)
ax.set_facecolor(BG)
ax.set_xlim(0, 20)
ax.set_ylim(0, 11)
ax.axis("off")

title    = ax.text(10, 10.4, "", color=TEXT,    fontsize=22, ha="center", va="center", fontweight="bold")
subtitle = ax.text(10, 9.85, "", color=SUBTEXT, fontsize=11, ha="center", va="center")

# ═══════════════════════════════════════════════════
# SECCIÓN 1 – Pregunta + Tokens  (x: 0.2 – 4.2)
# ═══════════════════════════════════════════════════
ax.text(2.1, 9.35, "Pregunta del Usuario", color=TEXT, fontsize=13, ha="center", fontweight="bold")

input_box = Rectangle((0.2, 7.9), 3.8, 1.2,
                       facecolor=GRAY, edgecolor=BORDER, lw=2, zorder=2)
ax.add_patch(input_box)
input_text_obj = ax.text(2.1, 8.5, "", color=TEXT, fontsize=9,
                         ha="center", va="center", zorder=3)

ax.text(2.1, 7.4, "Tokenización + Lematización",
        color=TEXT, fontsize=11, ha="center", fontweight="bold")

token_boxes, token_texts, token_pos = [], [], []
for i, tok in enumerate(tokens):
    x = 0.25 + i * 0.74
    b = Rectangle((x, 6.35), 0.66, 0.6,
                  facecolor=GRAY, edgecolor=BORDER, lw=1.3, zorder=2)
    ax.add_patch(b)
    token_boxes.append(b)
    cx, cy = x + 0.33, 6.65
    token_pos.append((cx, cy))
    t = ax.text(cx, cy, "", color=TEXT, fontsize=7.5,
                ha="center", va="center", zorder=3)
    token_texts.append(t)

# ═══════════════════════════════════════════════════
# SECCIÓN 2 – Bag of Words  (x ≈ 6.0)
# ═══════════════════════════════════════════════════
bow_x = 6.0
ax.text(bow_x, 9.35, "Bag of Words", color=TEXT, fontsize=13,
        ha="center", fontweight="bold")

bow_y = np.linspace(2.4, 8.8, 8)
bow_circles, bow_texts, bow_labels = [], [], []
for i, y in enumerate(bow_y):
    c = Circle((bow_x, y), 0.28, facecolor=GRAY, edgecolor=BORDER, lw=1.5, zorder=2)
    ax.add_patch(c)
    bow_circles.append(c)
    t = ax.text(bow_x, y, "", color=TEXT, fontsize=8.5,
                ha="center", va="center", fontweight="bold", zorder=3)
    bow_texts.append(t)
    lbl = ax.text(bow_x - 0.72, y, vocab_sample[i], color=SUBTEXT,
                  fontsize=6.5, ha="center", va="center", zorder=2)
    bow_labels.append(lbl)

# ═══════════════════════════════════════════════════
# SECCIÓN 3 – Capa Oculta 1 – 128  (x ≈ 9.2)
# ═══════════════════════════════════════════════════
h1_x = 9.2
ax.text(h1_x, 9.35, "Capa Oculta 1\n(128 neuronas)", color=TEXT, fontsize=12,
        ha="center", fontweight="bold")
ax.text(h1_x, 1.85, "ReLU + Dropout", color=SUBTEXT, fontsize=9,
        ha="center", fontweight="bold")

h1_y = np.linspace(2.4, 8.5, 6)
h1_circles, h1_texts = [], []
for y in h1_y:
    c = Circle((h1_x, y), 0.33, facecolor=GRAY, edgecolor=BORDER, lw=1.7, zorder=2)
    ax.add_patch(c)
    h1_circles.append(c)
    t = ax.text(h1_x, y, "", color=TEXT, fontsize=8,
                ha="center", va="center", fontweight="bold", zorder=3)
    h1_texts.append(t)

# ═══════════════════════════════════════════════════
# SECCIÓN 4 – Capa Oculta 2 – 64  (x ≈ 12.2)
# ═══════════════════════════════════════════════════
h2_x = 12.2
ax.text(h2_x, 9.35, "Capa Oculta 2\n(64 neuronas)", color=TEXT, fontsize=12,
        ha="center", fontweight="bold")
ax.text(h2_x, 2.3, "ReLU + Dropout", color=SUBTEXT, fontsize=9,
        ha="center", fontweight="bold")

h2_y = np.linspace(3.0, 8.0, 5)
h2_circles, h2_texts = [], []
for y in h2_y:
    c = Circle((h2_x, y), 0.33, facecolor=GRAY, edgecolor=BORDER, lw=1.7, zorder=2)
    ax.add_patch(c)
    h2_circles.append(c)
    t = ax.text(h2_x, y, "", color=TEXT, fontsize=8,
                ha="center", va="center", fontweight="bold", zorder=3)
    h2_texts.append(t)

# ═══════════════════════════════════════════════════
# SECCIÓN 5 – Output / Intents  (x ≈ 15.1)
# ═══════════════════════════════════════════════════
out_x = 15.1
ax.text(out_x, 9.35, "Clasificación\nde Intent", color=TEXT, fontsize=12,
        ha="center", fontweight="bold")
ax.text(out_x, 2.7, "Softmax", color=SUBTEXT, fontsize=9,
        ha="center", fontweight="bold")

out_y = np.linspace(3.5, 8.0, 5)
out_circles, out_texts = [], []
for i, y in enumerate(out_y):
    c = Circle((out_x, y), 0.35, facecolor=GRAY, edgecolor=BORDER, lw=1.8, zorder=2)
    ax.add_patch(c)
    out_circles.append(c)
    t = ax.text(out_x, y, "", color=TEXT, fontsize=8,
                ha="center", va="center", fontweight="bold", zorder=3)
    out_texts.append(t)
    col = GREEN if i == predicted_idx else SUBTEXT
    fw  = "bold" if i == predicted_idx else "normal"
    ax.text(out_x + 0.72, y, intents_short[i], color=col,
            fontsize=8, va="center", fontweight=fw)

# ═══════════════════════════════════════════════════
# SECCIÓN 6 – Respuesta  (x ≈ 18.5)
# ═══════════════════════════════════════════════════
ax.text(18.4, 9.35, "Respuesta", color=TEXT, fontsize=13,
        ha="center", fontweight="bold")

response_box = Rectangle((17.0, 5.8), 2.8, 3.0,
                          facecolor=GRAY, edgecolor=BORDER, lw=2, zorder=2)
ax.add_patch(response_box)
response_obj = ax.text(18.4, 7.3, "", color=TEXT, fontsize=9,
                       ha="center", va="center", zorder=3, linespacing=1.6)

intent_box = Rectangle((17.0, 5.0), 2.8, 0.65,
                        facecolor=GRAY, edgecolor=GREEN, lw=2, zorder=2)
ax.add_patch(intent_box)
intent_label = ax.text(18.4, 5.32, "", color=GREEN, fontsize=9,
                       ha="center", va="center", fontweight="bold", zorder=3)

# ── Conexiones estáticas ──
lines_tok_bow = []
for cx, cy in token_pos:
    for j, y2 in enumerate(bow_y):
        if bow_values[j] == 1:
            ln, = ax.plot([cx + 0.33, bow_x - 0.28], [cy, y2],
                          color=BLUE, lw=0.7, alpha=0.10, zorder=1)
            lines_tok_bow.append(ln)

lines_bow_h1 = []
for y1 in bow_y:
    for y2 in h1_y:
        ln, = ax.plot([bow_x + 0.28, h1_x - 0.33], [y1, y2],
                      color=BLUE, lw=0.5, alpha=0.08, zorder=1)
        lines_bow_h1.append(ln)

lines_h1_h2 = []
for y1 in h1_y:
    for y2 in h2_y:
        ln, = ax.plot([h1_x + 0.33, h2_x - 0.33], [y1, y2],
                      color=BLUE, lw=0.6, alpha=0.10, zorder=1)
        lines_h1_h2.append(ln)

lines_h2_out = []
for y1 in h2_y:
    for y2 in out_y:
        ln, = ax.plot([h2_x + 0.33, out_x - 0.35], [y1, y2],
                      color=BLUE, lw=0.7, alpha=0.11, zorder=1)
        lines_h2_out.append(ln)

line_out_resp, = ax.plot([out_x + 0.35, 17.0], [out_y[predicted_idx], 6.3],
                         color=BLUE, lw=1.5, alpha=0.2, zorder=1)

# ── Textos inferiores ──
formula_text = ax.text(10, 1.1, "", color=SUBTEXT, fontsize=12,
                       ha="center", va="center", fontweight="bold")
result_text  = ax.text(10, 0.45, "", color=GREEN, fontsize=16,
                       ha="center", va="center", fontweight="bold")

# ── Partículas ──
particles = []
for _ in range(30):
    p, = ax.plot([], [], "o", color=GREEN, markersize=4, alpha=0, zorder=5)
    particles.append(p)


# ── Helpers ──
def rgba(hex_color, alpha):
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i+2], 16) / 255 for i in (0, 2, 4))
    return (r, g, b, alpha)


def move_particle(artist, start, end, t, color=GREEN):
    t = max(0.0, min(1.0, t))
    x = start[0] + (end[0] - start[0]) * t
    y = start[1] + (end[1] - start[1]) * t
    artist.set_data([x], [y])
    artist.set_alpha(1.0)
    artist.set_color(color)


def reset_scene():
    title.set_text("")
    subtitle.set_text("")
    formula_text.set_text("")
    result_text.set_text("")
    input_text_obj.set_text("")

    input_box.set_facecolor(GRAY)
    input_box.set_edgecolor(BORDER)

    for b in token_boxes:
        b.set_facecolor(GRAY); b.set_edgecolor(BORDER)
    for t in token_texts:
        t.set_text("")

    for c in bow_circles:
        c.set_facecolor(GRAY); c.set_edgecolor(BORDER); c.set_linewidth(1.5)
    for t in bow_texts:
        t.set_text("")

    for c in h1_circles:
        c.set_facecolor(GRAY); c.set_edgecolor(BORDER); c.set_linewidth(1.7)
    for t in h1_texts:
        t.set_text("")

    for c in h2_circles:
        c.set_facecolor(GRAY); c.set_edgecolor(BORDER); c.set_linewidth(1.7)
    for t in h2_texts:
        t.set_text("")

    for c in out_circles:
        c.set_facecolor(GRAY); c.set_edgecolor(BORDER); c.set_linewidth(1.8)
    for t in out_texts:
        t.set_text("")

    for ln in lines_tok_bow:
        ln.set_color(BLUE); ln.set_alpha(0.10); ln.set_linewidth(0.7)
    for ln in lines_bow_h1:
        ln.set_color(BLUE); ln.set_alpha(0.08); ln.set_linewidth(0.5)
    for ln in lines_h1_h2:
        ln.set_color(BLUE); ln.set_alpha(0.10); ln.set_linewidth(0.6)
    for ln in lines_h2_out:
        ln.set_color(BLUE); ln.set_alpha(0.11); ln.set_linewidth(0.7)
    line_out_resp.set_color(BLUE); line_out_resp.set_alpha(0.2); line_out_resp.set_linewidth(1.5)

    response_box.set_facecolor(GRAY); response_box.set_edgecolor(BORDER); response_box.set_linewidth(2)
    response_obj.set_text("")
    intent_box.set_facecolor(GRAY); intent_box.set_edgecolor(GREEN)
    intent_label.set_text("")

    for p in particles:
        p.set_alpha(0)


# ── Animación ──
FRAMES = 265


def update(frame):
    reset_scene()

    # ─── FASE 0: Intro ───────────────────────────────────────
    if frame < 25:
        title.set_text("AI Chatbot con Redes Neuronales")
        subtitle.set_text(
            "Pregunta  →  Tokenización  →  Bag of Words  →  Red Neuronal  →  Intent  →  Respuesta")
        formula_text.set_text(
            "Clasificador neuronal entrenado con PyTorch para responder preguntas sobre Deep Learning")

    # ─── FASE 1: Input del usuario ───────────────────────────
    elif frame < 60:
        title.set_text("1. Entrada del Usuario")
        subtitle.set_text("El chatbot recibe la pregunta en texto libre")
        progress = min(1.0, (frame - 25) / 22)
        input_box.set_facecolor(rgba(BLUE, 0.13 + 0.12 * progress))
        input_box.set_edgecolor(BLUE)
        n_chars = max(1, int(len(user_question) * progress))
        cursor  = "_" if progress < 1.0 else ""
        input_text_obj.set_text(user_question[:n_chars] + cursor)
        formula_text.set_text(f'Usuario: "{user_question}"')

    # ─── FASE 2: Tokenización ────────────────────────────────
    elif frame < 100:
        title.set_text("2. Tokenización y Lematización")
        subtitle.set_text(
            "La pregunta se divide en tokens y se normalizan (lemmatize) las palabras")
        input_box.set_facecolor(rgba(BLUE, 0.15))
        input_box.set_edgecolor(BLUE)
        input_text_obj.set_text(user_question)

        n_tok = min(5, max(0, (frame - 60) // 8 + 1))
        for i in range(n_tok):
            token_boxes[i].set_facecolor(rgba(GREEN, 0.30))
            token_boxes[i].set_edgecolor(GREEN)
            token_texts[i].set_text(tokens[i])
            t_prog = min(1.0, (frame - 60 - i * 8) / 6)
            if t_prog > 0:
                move_particle(particles[i],
                              (input_box.get_x() + 1.9, 7.9),
                              token_pos[i], t_prog)

        formula_text.set_text(
            "nltk.word_tokenize()  →  WordNetLemmatizer().lemmatize(word.lower())")

    # ─── FASE 3: Bag of Words ────────────────────────────────
    elif frame < 145:
        title.set_text("3. Bag of Words")
        subtitle.set_text(
            "Cada palabra del vocabulario toma 1 si está presente en la pregunta, 0 si no")
        input_box.set_facecolor(rgba(BLUE, 0.08))
        input_text_obj.set_text(user_question)
        for i in range(5):
            token_boxes[i].set_facecolor(rgba(GREEN, 0.20))
            token_boxes[i].set_edgecolor(GREEN)
            token_texts[i].set_text(tokens[i])

        for ln in lines_tok_bow:
            ln.set_color(GREEN); ln.set_alpha(0.35); ln.set_linewidth(1.1)

        n_bow = min(8, max(0, (frame - 100) // 5 + 1))
        for i in range(n_bow):
            val = bow_values[i]
            if val == 1:
                bow_circles[i].set_facecolor(rgba(GREEN, 0.75))
                bow_circles[i].set_edgecolor(GREEN)
                bow_circles[i].set_linewidth(2.2)
            else:
                bow_circles[i].set_facecolor(rgba(RED, 0.28))
                bow_circles[i].set_edgecolor(RED)
                bow_circles[i].set_linewidth(1.5)
            bow_texts[i].set_text(str(val))

        for j in range(8):
            if bow_values[j] == 1:
                t_p = min(1.0, (frame - 100) / 28)
                move_particle(particles[j], token_pos[0], (bow_x, bow_y[j]), t_p)

        formula_text.set_text(
            "bag = [1 if word in tokens else 0  for word in vocabulary]  →  [0,1,0,0,0,1,0,1,…]")

    # ─── FASE 4: Capa Oculta 1 (128) ─────────────────────────
    elif frame < 185:
        title.set_text("4. Capa Oculta 1 — 128 Neuronas + ReLU")
        subtitle.set_text(
            "fc1: Linear(vocab→128) transforma el vector BoW en 128 activaciones no-lineales")

        input_text_obj.set_text(user_question)
        input_box.set_facecolor(rgba(BLUE, 0.05))
        for i in range(5):
            token_boxes[i].set_facecolor(rgba(GREEN, 0.13))
            token_texts[i].set_text(tokens[i])
        for i, val in enumerate(bow_values):
            col = GREEN if val else RED
            bow_circles[i].set_facecolor(rgba(col, 0.50 if val else 0.18))
            bow_circles[i].set_edgecolor(col)
            bow_texts[i].set_text(str(val))

        for ln in lines_bow_h1:
            ln.set_color(GREEN); ln.set_alpha(0.22); ln.set_linewidth(0.8)

        progress = min(1.0, (frame - 145) / 28)
        for i, v in enumerate(hidden1_vals):
            act = v * progress
            h1_circles[i].set_facecolor(rgba(GREEN, 0.22 + 0.68 * min(act, 1.0)))
            h1_texts[i].set_text(f"{act:.2f}")

        for i in range(6):
            src_y = bow_y[i % 8]
            move_particle(particles[8 + i],
                          (bow_x, src_y), (h1_x, h1_y[i]),
                          min(1.0, (frame - 145) / 22))

        formula_text.set_text(
            "x = ReLU( fc1(bag_of_words) )   →   Dropout(p=0.5)   →   128 activaciones")

    # ─── FASE 5: Capa Oculta 2 (64) + Output ─────────────────
    elif frame < 230:
        title.set_text("5. Capa Oculta 2 (64) → Softmax → Intent")
        subtitle.set_text(
            "fc2: 128→64  |  fc3: 64→n_intents  |  Softmax convierte logits en probabilidades")

        input_text_obj.set_text(user_question)
        input_box.set_facecolor(rgba(BLUE, 0.04))
        for i in range(5):
            token_boxes[i].set_facecolor(rgba(GREEN, 0.09))
            token_texts[i].set_text(tokens[i])
        for i, val in enumerate(bow_values):
            col = GREEN if val else RED
            bow_circles[i].set_facecolor(rgba(col, 0.35 if val else 0.12))
            bow_circles[i].set_edgecolor(col)
            bow_texts[i].set_text(str(val))
        for i, v in enumerate(hidden1_vals):
            h1_circles[i].set_facecolor(rgba(GREEN, 0.22 + 0.68 * min(v, 1.0)))
            h1_texts[i].set_text(f"{v:.2f}")

        for ln in lines_h1_h2:
            ln.set_color(GREEN); ln.set_alpha(0.28); ln.set_linewidth(0.9)

        prog2 = min(1.0, (frame - 185) / 22)
        for i, v in enumerate(hidden2_vals):
            act = v * prog2
            h2_circles[i].set_facecolor(rgba(GREEN, 0.22 + 0.68 * min(act, 1.0)))
            h2_texts[i].set_text(f"{act:.2f}")

        for i in range(5):
            move_particle(particles[14 + i],
                          (h1_x, h1_y[i % 6]), (h2_x, h2_y[i]),
                          min(1.0, (frame - 185) / 18))

        prog_out = min(1.0, max(0, (frame - 207)) / 15)
        for ln in lines_h2_out:
            ln.set_color(GREEN); ln.set_alpha(0.25); ln.set_linewidth(0.9)
        for i, prob in enumerate(output_probs):
            col = PURPLE if i == predicted_idx else BLUE
            out_circles[i].set_facecolor(rgba(col, 0.18 + 0.72 * min(prob * prog_out * 3, 1.0)))
            out_texts[i].set_text(f"{prob * prog_out:.2f}")

        for i in range(5):
            col_p = PURPLE if i == predicted_idx else GREEN
            move_particle(particles[20 + i],
                          (h2_x, h2_y[i]), (out_x, out_y[i]),
                          min(1.0, max(0, (frame - 207)) / 14),
                          color=col_p)

        formula_text.set_text(
            "x = ReLU(fc2(x)) → fc3(x) → Softmax → [0.04, 0.08, 0.81, 0.05, 0.02]")

    # ─── FASE 6: Resultado y Respuesta ───────────────────────
    else:
        title.set_text("6. Intent Detectado → Respuesta del Chatbot")
        subtitle.set_text(
            f"El intent con mayor probabilidad: \"{predicted_tag}\" → respuesta educativa")

        input_text_obj.set_text(user_question)
        input_box.set_facecolor(rgba(BLUE, 0.04))

        for i in range(5):
            token_boxes[i].set_facecolor(rgba(GREEN, 0.09))
            token_texts[i].set_text(tokens[i])
        for i, val in enumerate(bow_values):
            col = GREEN if val else RED
            bow_circles[i].set_facecolor(rgba(col, 0.30 if val else 0.10))
            bow_circles[i].set_edgecolor(col)
            bow_texts[i].set_text(str(val))
        for i, v in enumerate(hidden1_vals):
            h1_circles[i].set_facecolor(rgba(GREEN, 0.22 + 0.68 * min(v, 1.0)))
            h1_texts[i].set_text(f"{v:.2f}")
        for i, v in enumerate(hidden2_vals):
            h2_circles[i].set_facecolor(rgba(GREEN, 0.22 + 0.68 * min(v, 1.0)))
            h2_texts[i].set_text(f"{v:.2f}")

        for i, prob in enumerate(output_probs):
            if i == predicted_idx:
                pulse = 0.55 + 0.45 * abs(np.sin((frame - 230) * 0.30))
                out_circles[i].set_facecolor(rgba(GREEN, pulse))
                out_circles[i].set_edgecolor("white")
                out_circles[i].set_linewidth(3.5)
            else:
                out_circles[i].set_facecolor(rgba(BLUE, 0.18))
                out_circles[i].set_edgecolor(BORDER)
            out_texts[i].set_text(f"{prob:.2f}")

        line_out_resp.set_color(GREEN); line_out_resp.set_alpha(0.65); line_out_resp.set_linewidth(2.5)

        response_box.set_facecolor(rgba(GREEN, 0.08))
        response_box.set_edgecolor(GREEN)
        response_box.set_linewidth(2.5)
        response_obj.set_text(response_text)

        intent_box.set_facecolor(rgba(GREEN, 0.20))
        intent_box.set_edgecolor(GREEN)
        intent_label.set_text(f"Intent: {predicted_tag}")

        formula_text.set_text(
            f"torch.argmax(predictions) = {predicted_idx}  →  \"{predicted_tag}\"  (81%)")
        result_text.set_text(
            "Chatbot responde sobre: neurona artificial  |  Probabilidad: 81%")

    return []


ani = FuncAnimation(fig, update, frames=FRAMES, interval=95, blit=False, repeat=True)

plt.tight_layout(pad=1.5)

script_dir  = os.path.dirname(os.path.abspath(__file__))
output_path = os.path.join(script_dir, "chatbot_nn_animation.gif")

print("Guardando animación... (puede tardar un minuto)")
ani.save(output_path, writer="pillow", fps=12)
print(f"GIF guardado: {output_path}")

plt.show()

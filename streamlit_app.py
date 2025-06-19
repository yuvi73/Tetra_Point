import streamlit as st
import numpy as np
import plotly.graph_objects as go

# --- Tetralemma Logic ---
POLARITIES = {
    'EXPRESSED': 1,
    'SUPPRESSED': 0,
    'INAPPLICABLE': -1,
    'EMPTY': -2
}
POLARITY_LABELS = ['a', '¬a', 'a∧¬a', '¬(a∨¬a)']
POLARITY_COLORS = {
    1: 'red',
    0: 'blue',
    -1: 'yellow',
    -2: 'black'
}

# Tetrapoint logic

def create_tetrapoint(a, not_a, both, neither):
    return np.array([a, not_a, both, neither])

def negation_transform(tetrapoint):
    return np.array([
        tetrapoint[1],
        tetrapoint[2],
        tetrapoint[3],
        tetrapoint[0]
    ])

def contradiction_product(t1, t2):
    def polar_product(p1, p2):
        if p1 == -2 or p2 == -2: return -2
        if p1 == -1 and p2 == -1: return -1
        if p1 == -1 or p2 == -1: return -1
        if p1 == 0 or p2 == 0: return 0
        if p1 == 1 and p2 == 1: return 1
        return 0
    return np.array([
        polar_product(t1[0], t2[0]),
        polar_product(t1[1], t2[1]),
        polar_product(t1[2], t2[2]),
        polar_product(t1[3], t2[3])
    ])

def is_empty(t):
    return np.all(t == -2)

def map_tetrapoint_to_cube(tetrapoint):
    x = (tetrapoint[0] + 2) / 4
    y = (tetrapoint[1] + 2) / 4
    z = (tetrapoint[2] + 2) / 4
    diagonal = (tetrapoint[3] + 2) / 4
    x = (x + diagonal) / 2
    y = (y + diagonal) / 2
    z = (z + diagonal) / 2
    return np.array([x, y, z])

# --- Streamlit UI ---
st.set_page_config(page_title="Tetralemma Contradiction Explorer", layout="wide")
st.title("🧠 Tetralemma Contradiction Explorer")
st.markdown("""
Interactively explore the logic of contradiction, negation, and emptiness using the Tetralemma Space (𝕋).
- Input a proposition and set its four polarities
- Cycle through negation (τ)
- Fuse two tetrapoints (⊗)
- Visualize the state in 3D
- See the emptiness limit
""")

# --- Input Panel ---
st.sidebar.header("Tetrapoint Input")
prop = st.sidebar.text_input("Proposition (for your reference)", "The cat is on the mat")

# Simplified polarity selection
polarity_options = [1, 0, -1, -2]
polarity_names = ['EXPRESSED', 'SUPPRESSED', 'INAPPLICABLE', 'EMPTY']

def polarity_select(label, default):
    options = [f"{val} ({name})" for val, name in zip(polarity_options, polarity_names)]
    selected = st.sidebar.selectbox(label, options, index=polarity_options.index(default))
    return polarity_options[options.index(selected)]

a = polarity_select("a (Affirmation)", 1)
not_a = polarity_select("¬a (Negation)", 0)
both = polarity_select("a∧¬a (Both)", 0)
neither = polarity_select("¬(a∨¬a) (Neither)", 0)

current = create_tetrapoint(a, not_a, both, neither)

# --- Negation Cycle ---
st.sidebar.header("Negation Cycle (τ)")
neg_steps = st.sidebar.slider("Steps to cycle τ", 0, 12, 0)
neg_t = current.copy()
for _ in range(neg_steps):
    neg_t = negation_transform(neg_t)

# --- Contradiction Fusion ---
st.sidebar.header("Contradiction Fusion (⊗)")
fuse = st.sidebar.checkbox("Fuse with another tetrapoint?")
if fuse:
    a2 = polarity_select("a₂ (Affirmation)", 1)
    not_a2 = polarity_select("¬a₂ (Negation)", 0)
    both2 = polarity_select("a₂∧¬a₂ (Both)", 0)
    neither2 = polarity_select("¬(a₂∨¬a₂) (Neither)", 0)
    t2 = create_tetrapoint(a2, not_a2, both2, neither2)
    fused = contradiction_product(neg_t, t2)
else:
    fused = neg_t

# --- Emptiness Limit ---
st.sidebar.header("Emptiness Limit")
lim_steps = st.sidebar.slider("τ iterations for emptiness limit", 1, 50, 10)
lim_t = fused.copy()
for _ in range(lim_steps):
    lim_t = negation_transform(lim_t)
    if is_empty(lim_t):
        break

# --- 3D Visualization ---
def plot_tetrapoint_3d(t, label, color):
    pos = map_tetrapoint_to_cube(t)
    return go.Scatter3d(
        x=[pos[0]], y=[pos[1]], z=[pos[2]],
        mode='markers+text',
        marker=dict(size=16, color=color, line=dict(color='black', width=2)),
        text=[label], textposition="top center"
    )

cube_lines = [
    ([0,1],[0,0],[0,0]), ([1,1],[0,1],[0,0]), ([1,0],[1,1],[0,0]), ([0,0],[1,0],[0,0]), # bottom
    ([0,1],[0,0],[1,1]), ([1,1],[0,1],[1,1]), ([1,0],[1,1],[1,1]), ([0,0],[1,0],[1,1]), # top
    ([0,0],[0,0],[0,1]), ([1,1],[0,0],[0,1]), ([1,1],[1,1],[0,1]), ([0,0],[1,1],[0,1])  # sides
]

fig = go.Figure()
# Draw cube
for x, y, z in cube_lines:
    fig.add_trace(go.Scatter3d(x=x, y=y, z=z, mode='lines', line=dict(color='gray', width=2), showlegend=False))

# Plot current, negated, fused, and limit tetrapoints
fig.add_trace(plot_tetrapoint_3d(current, "Current", POLARITY_COLORS.get(current[0], 'gray')))
fig.add_trace(plot_tetrapoint_3d(neg_t, f"τ^{neg_steps}(Current)", POLARITY_COLORS.get(neg_t[0], 'gray')))
if fuse:
    fig.add_trace(plot_tetrapoint_3d(t2, "Fused With", POLARITY_COLORS.get(t2[0], 'gray')))
fig.add_trace(plot_tetrapoint_3d(lim_t, f"Emptiness Limit", POLARITY_COLORS.get(lim_t[0], 'gray')))

fig.update_layout(
    scene=dict(
        xaxis_title='a (Affirmation)',
        yaxis_title='¬a (Negation)',
        zaxis_title='a∧¬a (Both)',
        xaxis=dict(range=[0,1]),
        yaxis=dict(range=[0,1]),
        zaxis=dict(range=[0,1]),
    ),
    width=800, height=600,
    margin=dict(l=0, r=0, b=0, t=40),
    showlegend=False,
    title="Tetralemma Space (𝕋) - 3D Visualization"
)
st.plotly_chart(fig)

# --- State Table ---
def tetrapoint_str(t):
    return f"({t[0]}, {t[1]}, {t[2]}, {t[3]})"

st.subheader("Tetrapoint States")
st.write(f"**Current:** {tetrapoint_str(current)}")
st.write(f"**After τ^{neg_steps}:** {tetrapoint_str(neg_t)}")
if fuse:
    st.write(f"**Fused With:** {tetrapoint_str(t2)}")
st.write(f"**Emptiness Limit (after {lim_steps} τ):** {tetrapoint_str(lim_t)}")

# --- Interpretation Panel ---
def interpret_tetrapoint(t):
    if is_empty(t):
        return "**Emptiness (Ψ):** All conceptual ground is erased."
    if np.all(t == 1):
        return "**Total Affirmation:** All poles are expressed."
    if np.all(t == 0):
        return "**Total Suppression:** All poles are suppressed."
    if np.all(t == -1):
        return "**Inapplicable:** All poles are inapplicable."
    if t[0] == 1 and t[1] == 0 and t[2] == 0 and t[3] == 0:
        return "**Affirmation (P):** The proposition is affirmed."
    if t[0] == 0 and t[1] == 1 and t[2] == 0 and t[3] == 0:
        return "**Negation (¬P):** The proposition is denied."
    if t[0] == 0 and t[1] == 0 and t[2] == 1 and t[3] == 0:
        return "**Both (P∧¬P):** Both affirmation and negation are present."
    if t[0] == 0 and t[1] == 0 and t[2] == 0 and t[3] == 1:
        return "**Neither (¬(P∨¬P)):** Neither affirmation nor negation."
    return "**Mixed/Paradoxical:** This state is a unique blend of polarities."

st.subheader("Interpretation")
st.markdown(interpret_tetrapoint(lim_t))

st.info("This tool is a novel application of the Tetralemma Space (𝕋), allowing you to explore contradiction, negation, and emptiness interactively.") 
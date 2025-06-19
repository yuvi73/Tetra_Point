# Tetralemma Space (𝕋) - Mathematical Structure

A foundational mathematical structure that embodies the Catuṣkoṭi (Four-Cornered) logic from Madhyamaka Buddhism, treating contradiction as a generative relation rather than an error.

## 🧠 Concept

The Tetralemma Space (𝕋) is a new abstract mathematical structure where every proposition is a point with four-valued polarity and non-linear negation topology. It:

- **Treats contradiction as a generative relation** (not an error or boundary)
- **Allows truth to reside in indeterminacy**
- **Respects and manifests Nāgārjuna's logic of emptiness**
- **Forms a non-Cartesian, non-Boolean category**

## 🔢 Mathematical Definition

### Elements of 𝕋

Each tetrapoint t ∈ 𝕋 is a 4-tuple:
```
t = (a, ¬a, a∧¬a, ¬(a∨¬a))
```

Where each element can be:
- **1** (expressed)
- **0** (suppressed)  
- **Ø** (inapplicable)
- **Ψ** (empty)

### Internal Logic: T-Polarity Algebra

**Contradiction Product (⊗):**
```
t₁ ⊗ t₂ = element-wise conjunction of all 4 poles
```

**Tetralemma Morphism (τ):**
```
τ(a, ¬a, a∧¬a, ¬(a∨¬a)) = (¬a, a∧¬a, ¬(a∨¬a), a)
```

**Emptiness as Limit:**
```
lim(n→∞) τⁿ(t) = Ψ
```

## 🛠 Implementation

This repository contains both a C implementation and a Python visualization.

### C Implementation

The C implementation provides the core mathematical structure with:

- **Tetrapoint operations** (creation, negation transform, contradiction product)
- **Tetralemma Space management** (dynamic space with tetrapoints)
- **Emptiness limit calculation**
- **Comprehensive testing and demonstration**

#### Building and Running

```bash
# Compile the C implementation
make

# Run the demonstration
./tetralemma

# Or use make run
make run

# Clean build files
make clean
```

#### Example Output

```
🧠 Tetralemma Space (𝕋) - Mathematical Structure Implementation
=============================================================

Tetralemma Space (𝕋) - 4 points:
  Point 0: (1, 0, 0, 0)
  Point 1: (0, 1, 0, 0)
  Point 2: (0, 0, 1, 0)
  Point 3: (0, 0, 0, 1)

🌀 Negation Cycle (τ transformation):
Step 0: (1, 0, 0, 0) => P (Affirmation)
Step 1: (0, 0, 0, 1) => ¬(P ∨ ¬P) (Neither)
Step 2: (0, 0, 1, 0) => P ∧ ¬P (Both)
Step 3: (0, 1, 0, 0) => ¬P (Negation)
Step 4: (1, 0, 0, 0) => τ⁴(P)
Step 5: (0, 0, 0, 1) => τ⁵(P)
```

### Python Visualization

The Python visualization creates a 3D cube representation showing:

- **Tetrapoints as colored spheres** in 3D space
- **Negation cycles** as connected paths
- **Four-polarity system** with color coding
- **Animated negation transformations**

#### Setup and Running

```bash
# Install Python dependencies
pip install -r requirements.txt

# Run the visualization
python tetralemma_visualization.py
```

This will generate:
- `tetralemma_cube.png` - Static 3D cube visualization
- `tetralemma_cycle.gif` - Animated negation cycle

## 🎯 Key Features

### Novel Mathematical Properties

| Feature | Boolean Logic | Fuzzy Logic | Category Theory | Tetralemma Space (𝕋) |
|---------|---------------|-------------|-----------------|---------------------|
| Binary Truth | ✅ | ❌ | ❌ | ❌ |
| Contradiction as Failure | ✅ | ❌ | ❌ | ❌ |
| Contradiction as Process | ❌ | ❌ | ❌ | ✅ |
| Circular Negation | ❌ | ❌ | ❌ | ✅ |
| Emptiness as Limit | ❌ | ❌ | ❌ | ✅ |
| Non-idempotent morphisms | ❌ | ❌ | ✅ | ✅ |
| Philosophical Ground | ❌ | ❌ | ❌ | Madhyamaka 🕉 |

### Four-Valued Polarity System

1. **EXPRESSED (1)** - Red: Proposition is actively affirmed
2. **SUPPRESSED (0)** - Blue: Proposition is actively denied  
3. **INAPPLICABLE (Ø)** - Yellow: Proposition is neither affirmed nor denied
4. **EMPTY (Ψ)** - Black: Proposition has no conceptual ground

## 🔄 Negation Cycle

The τ transformation creates a cyclical dialectic:

```
P → ¬(P ∨ ¬P) → P ∧ ¬P → ¬P → P → ...
```

This demonstrates how contradiction moves the structure forward through a four-stage polarity cycle.

## 🧘‍♂️ Philosophical Foundation

This structure embodies the Madhyamaka principle of **śūnyatā** (emptiness) by:

- **Breaking down subject-object duality** through self-exhaustion of logic
- **Treating contradictions as generative** rather than destructive
- **Allowing truth to emerge from indeterminacy**
- **Providing a mathematical model** for Nāgārjuna's four-cornered logic

## 🚀 Future Applications

Potential uses for this mathematical structure:

- **Buddhist AI systems** with contradiction tolerance
- **Poetry generators** based on tetralemma logic
- **Theorem provers** that handle paradoxes gracefully
- **Philosophical reasoning engines**
- **Creative problem-solving systems**

## 📚 References

- Nāgārjuna's *Mūlamadhyamakakārikā* (Fundamental Verses on the Middle Way)
- Catuṣkoṭi (Four-Cornered) logic
- Madhyamaka philosophy of emptiness
- Non-classical logics and paraconsistent reasoning

## 🎉 License

This project is open source and available under the MIT License.

---

*"All phenomena are empty of inherent existence" - Nāgārjuna* 
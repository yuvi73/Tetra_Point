#!/usr/bin/env python3
"""
Tetralemma Space (𝕋) Visualization
==================================

A 3D cube visualization of the Tetralemma Space mathematical structure,
showing the four polarities and their cyclical transformations.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.animation as animation
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches

class TetralemmaVisualizer:
    def __init__(self):
        self.fig = plt.figure(figsize=(15, 10))
        self.ax = self.fig.add_subplot(111, projection='3d')
        
        # Define the four polarities
        self.polarities = {
            'EXPRESSED': 1,
            'SUPPRESSED': 0, 
            'INAPPLICABLE': -1,
            'EMPTY': -2
        }
        
        # Color mapping for polarities
        self.polarity_colors = {
            1: 'red',      # EXPRESSED
            0: 'blue',     # SUPPRESSED
            -1: 'yellow',  # INAPPLICABLE
            -2: 'black'    # EMPTY
        }
        
        # Cube vertices (8 corners of a unit cube)
        self.cube_vertices = np.array([
            [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
            [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]
        ])
        
        # Cube faces (6 faces of the cube)
        self.cube_faces = [
            [0, 1, 2, 3],  # bottom
            [4, 5, 6, 7],  # top
            [0, 1, 5, 4],  # front
            [2, 3, 7, 6],  # back
            [1, 2, 6, 5],  # right
            [0, 3, 7, 4]   # left
        ]
        
        # Tetrapoint structure: (a, ¬a, a∧¬a, ¬(a∨¬a))
        self.tetrapoint_labels = ['a', '¬a', 'a∧¬a', '¬(a∨¬a)']
        
    def create_tetrapoint(self, a, not_a, both, neither):
        """Create a tetrapoint with the four polarities"""
        return np.array([a, not_a, both, neither])
    
    def negation_transform(self, tetrapoint):
        """Apply the negation transform τ: cyclical permutation"""
        return np.array([
            tetrapoint[1],  # ¬a becomes a
            tetrapoint[2],  # a∧¬a becomes ¬a
            tetrapoint[3],  # ¬(a∨¬a) becomes a∧¬a
            tetrapoint[0]   # a becomes ¬(a∨¬a)
        ])
    
    def contradiction_product(self, t1, t2):
        """Apply contradiction product ⊗ between two tetrapoints"""
        def polar_product(p1, p2):
            if p1 == -2 or p2 == -2: return -2  # EMPTY
            if p1 == -1 and p2 == -1: return -1  # INAPPLICABLE
            if p1 == -1 or p2 == -1: return -1  # INAPPLICABLE
            if p1 == 0 or p2 == 0: return 0  # SUPPRESSED
            if p1 == 1 and p2 == 1: return 1  # EXPRESSED
            return 0  # Default to SUPPRESSED
        
        return np.array([
            polar_product(t1[0], t2[0]),
            polar_product(t1[1], t2[1]),
            polar_product(t1[2], t2[2]),
            polar_product(t1[3], t2[3])
        ])
    
    def map_tetrapoint_to_cube(self, tetrapoint):
        """Map a tetrapoint to cube coordinates based on polarities"""
        # Map each polarity to a coordinate axis
        # a -> x, ¬a -> y, a∧¬a -> z, ¬(a∨¬a) -> diagonal influence
        x = (tetrapoint[0] + 2) / 4  # Normalize to [0,1]
        y = (tetrapoint[1] + 2) / 4
        z = (tetrapoint[2] + 2) / 4
        
        # Apply diagonal influence from the fourth polarity
        diagonal_influence = (tetrapoint[3] + 2) / 4
        x = (x + diagonal_influence) / 2
        y = (y + diagonal_influence) / 2
        z = (z + diagonal_influence) / 2
        
        return np.array([x, y, z])
    
    def draw_cube(self, center=np.array([0.5, 0.5, 0.5]), size=1.0, alpha=0.1):
        """Draw the base cube structure"""
        vertices = self.cube_vertices * size + center - size/2
        
        # Create faces
        faces = []
        for face in self.cube_faces:
            face_vertices = [vertices[i] for i in face]
            faces.append(face_vertices)
        
        # Draw cube
        poly3d = Poly3DCollection(faces, alpha=alpha, facecolor='gray', edgecolor='black')
        self.ax.add_collection3d(poly3d)
    
    def plot_tetrapoint(self, tetrapoint, position, label="", size=100):
        """Plot a tetrapoint as a colored sphere"""
        color = self.polarity_colors.get(tetrapoint[0], 'gray')  # Use first polarity for color
        
        # Plot the point
        self.ax.scatter(position[0], position[1], position[2], 
                       c=color, s=size, alpha=0.8, edgecolors='black')
        
        # Add label
        if label:
            self.ax.text(position[0], position[1], position[2] + 0.1, 
                        label, fontsize=8, ha='center')
    
    def plot_negation_cycle(self, initial_tetrapoint, steps=4):
        """Plot the negation cycle transformation"""
        current = initial_tetrapoint.copy()
        positions = []
        tetrapoints = [current.copy()]
        
        for i in range(steps + 1):
            pos = self.map_tetrapoint_to_cube(current)
            positions.append(pos)
            
            if i < steps:
                current = self.negation_transform(current)
                tetrapoints.append(current.copy())
        
        # Plot the cycle
        positions = np.array(positions)
        self.ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], 
                    'k--', alpha=0.5, linewidth=2)
        
        # Plot each point in the cycle
        labels = ['P', '¬(P∨¬P)', 'P∧¬P', '¬P', 'τ⁴(P)']
        for i, (pos, tetrapoint) in enumerate(zip(positions, tetrapoints)):
            self.plot_tetrapoint(tetrapoint, pos, labels[i] if i < len(labels) else f"τ^{i}(P)")
    
    def create_legend(self):
        """Create a legend for the polarity colors"""
        legend_elements = []
        for polarity, color in self.polarity_colors.items():
            if polarity == 1:
                label = "EXPRESSED (1)"
            elif polarity == 0:
                label = "SUPPRESSED (0)"
            elif polarity == -1:
                label = "INAPPLICABLE (Ø)"
            else:
                label = "EMPTY (Ψ)"
            
            legend_elements.append(mpatches.Patch(color=color, label=label))
        
        self.ax.legend(handles=legend_elements, loc='upper right')
    
    def visualize_tetralemma_space(self):
        """Main visualization function"""
        self.ax.clear()
        
        # Set up the plot
        self.ax.set_xlabel('a (Affirmation)')
        self.ax.set_ylabel('¬a (Negation)')
        self.ax.set_zlabel('a∧¬a (Both)')
        self.ax.set_title('Tetralemma Space (𝕋) - 3D Cube Visualization\nFour-Valued Polarity System', 
                         fontsize=14, fontweight='bold')
        
        # Draw base cube
        self.draw_cube()
        
        # Create example tetrapoints
        examples = [
            self.create_tetrapoint(1, 0, 0, 0),   # P
            self.create_tetrapoint(0, 1, 0, 0),   # ¬P
            self.create_tetrapoint(0, 0, 1, 0),   # P ∧ ¬P
            self.create_tetrapoint(0, 0, 0, 1),   # ¬(P ∨ ¬P)
            self.create_tetrapoint(1, 1, 0, 0),   # Both P and ¬P expressed
            self.create_tetrapoint(-1, 0, 0, 0),  # Inapplicable
            self.create_tetrapoint(-2, -2, -2, -2) # Empty
        ]
        
        example_labels = ['P', '¬P', 'P∧¬P', '¬(P∨¬P)', 'P+¬P', 'Ø', 'Ψ']
        
        # Plot example points
        for tetrapoint, label in zip(examples, example_labels):
            pos = self.map_tetrapoint_to_cube(tetrapoint)
            self.plot_tetrapoint(tetrapoint, pos, label)
        
        # Plot negation cycle starting from P
        self.plot_negation_cycle(examples[0], 4)
        
        # Create legend
        self.create_legend()
        
        # Set view
        self.ax.view_init(elev=20, azim=45)
        self.ax.set_xlim(0, 1)
        self.ax.set_ylim(0, 1)
        self.ax.set_zlim(0, 1)
        
        plt.tight_layout()
    
    def animate_negation_cycle(self, frames=50):
        """Create an animation of the negation cycle"""
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        
        initial = self.create_tetrapoint(1, 0, 0, 0)  # Start with P
        cycle_data = []
        current = initial.copy()
        
        # Generate cycle data
        for i in range(frames):
            cycle_data.append(current.copy())
            current = self.negation_transform(current)
        
        def animate(frame):
            ax.clear()
            ax.set_xlim(-2.5, 2.5)
            ax.set_ylim(-2.5, 2.5)
            ax.set_title(f'Tetralemma Negation Cycle - Frame {frame}', fontsize=12)
            ax.set_xlabel('Polarity Value')
            ax.set_ylabel('Component')
            
            tetrapoint = cycle_data[frame]
            components = ['a', '¬a', 'a∧¬a', '¬(a∨¬a)']
            colors = [self.polarity_colors.get(val, 'gray') for val in tetrapoint]
            
            bars = ax.bar(components, tetrapoint, color=colors, alpha=0.7)
            ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            ax.grid(True, alpha=0.3)
            
            # Add value labels on bars
            for bar, val in zip(bars, tetrapoint):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{val}', ha='center', va='bottom' if height >= 0 else 'top')
        
        anim = animation.FuncAnimation(fig, animate, frames=frames, 
                                     interval=200, repeat=True)
        return anim

def main():
    """Main function to run the visualization"""
    print("🧠 Tetralemma Space (𝕋) - 3D Cube Visualization")
    print("=" * 50)
    
    # Create visualizer
    viz = TetralemmaVisualizer()
    
    # Create static visualization
    print("Creating 3D cube visualization...")
    viz.visualize_tetralemma_space()
    
    # Save the static plot
    plt.savefig('tetralemma_cube.png', dpi=300, bbox_inches='tight')
    print("✅ Static visualization saved as 'tetralemma_cube.png'")
    
    # Show the plot
    plt.show()
    
    # Create animation
    print("\nCreating negation cycle animation...")
    anim = viz.animate_negation_cycle()
    
    # Save animation
    anim.save('tetralemma_cycle.gif', writer='pillow', fps=5)
    print("✅ Animation saved as 'tetralemma_cycle.gif'")
    
    # Show animation
    plt.show()
    
    print("\n🎉 Visualization complete!")
    print("The cube shows how tetrapoints move through the four-polarity space.")
    print("The negation cycle demonstrates the cyclical transformation τ.")
    print("Colors represent the four polarities: EXPRESSED (red), SUPPRESSED (blue),")
    print("INAPPLICABLE (yellow), and EMPTY (black).")

if __name__ == "__main__":
    main() 
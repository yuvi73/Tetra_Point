#include "tetralemma.h"

// Helper function for polar product
static Polarity polar_product(Polarity p1, Polarity p2) {
    if (p1 == EMPTY || p2 == EMPTY) return EMPTY;
    if (p1 == INAPPLICABLE && p2 == INAPPLICABLE) return INAPPLICABLE;
    if (p1 == INAPPLICABLE || p2 == INAPPLICABLE) return INAPPLICABLE;
    if (p1 == SUPPRESSED || p2 == SUPPRESSED) return SUPPRESSED;
    if (p1 == EXPRESSED && p2 == EXPRESSED) return EXPRESSED;
    return SUPPRESSED; // Default case
}

// Tetrapoint operations

Tetrapoint tetrapoint_create(Polarity a, Polarity not_a, Polarity both, Polarity neither) {
    Tetrapoint t = {a, not_a, both, neither};
    return t;
}

char* polarity_to_string(Polarity p) {
    switch(p) {
        case EXPRESSED: return "1";
        case SUPPRESSED: return "0";
        case INAPPLICABLE: return "Ø";
        case EMPTY: return "Ψ";
        default: return "?";
    }
}

void tetrapoint_print(const Tetrapoint* t) {
    printf("(%s, %s, %s, %s)", 
           polarity_to_string(t->a),
           polarity_to_string(t->not_a),
           polarity_to_string(t->both),
           polarity_to_string(t->neither));
}

// Tetralemma Morphism (τ): cyclical permutation of polarities
// τ(a, ¬a, a∧¬a, ¬(a∨¬a)) = (¬a, a∧¬a, ¬(a∨¬a), a)
Tetrapoint tetrapoint_negation_transform(const Tetrapoint* t) {
    Tetrapoint result = {
        t->not_a,      // ¬a becomes a
        t->both,       // a∧¬a becomes ¬a
        t->neither,    // ¬(a∨¬a) becomes a∧¬a
        t->a           // a becomes ¬(a∨¬a)
    };
    return result;
}

// Contradiction Product (⊗): element-wise conjunction of all 4 poles
// This is not boolean AND, but a polar product that produces collapse or intensification
Tetrapoint tetrapoint_contradiction_product(const Tetrapoint* t1, const Tetrapoint* t2) {
    Tetrapoint result;
    
    // Define polar product rules
    // 1 ⊗ 1 = 1 (intensification)
    // 1 ⊗ 0 = 0 (suppression)
    // 1 ⊗ Ø = Ø (inapplicable)
    // 1 ⊗ Ψ = Ψ (empty)
    // 0 ⊗ 0 = 0 (suppression)
    // 0 ⊗ Ø = 0 (suppression)
    // 0 ⊗ Ψ = 0 (suppression)
    // Ø ⊗ Ø = Ø (inapplicable)
    // Ø ⊗ Ψ = Ψ (empty)
    // Ψ ⊗ Ψ = Ψ (empty)
    
    result.a = polar_product(t1->a, t2->a);
    result.not_a = polar_product(t1->not_a, t2->not_a);
    result.both = polar_product(t1->both, t2->both);
    result.neither = polar_product(t1->neither, t2->neither);
    
    return result;
}

int tetrapoint_is_empty(const Tetrapoint* t) {
    return (t->a == EMPTY && t->not_a == EMPTY && 
            t->both == EMPTY && t->neither == EMPTY);
}

int tetrapoint_equals(const Tetrapoint* t1, const Tetrapoint* t2) {
    return (t1->a == t2->a && t1->not_a == t2->not_a && 
            t1->both == t2->both && t1->neither == t2->neither);
}

// Tetralemma Space operations

TetralemmaSpace* tetralemma_space_create(int initial_capacity) {
    TetralemmaSpace* space = malloc(sizeof(TetralemmaSpace));
    if (!space) return NULL;
    
    space->points = malloc(initial_capacity * sizeof(Tetrapoint));
    if (!space->points) {
        free(space);
        return NULL;
    }
    
    space->capacity = initial_capacity;
    space->size = 0;
    return space;
}

void tetralemma_space_destroy(TetralemmaSpace* space) {
    if (space) {
        free(space->points);
        free(space);
    }
}

int tetralemma_space_add(TetralemmaSpace* space, const Tetrapoint* t) {
    if (!space || !t) return 0;
    
    if (space->size >= space->capacity) {
        int new_capacity = space->capacity * 2;
        Tetrapoint* new_points = realloc(space->points, new_capacity * sizeof(Tetrapoint));
        if (!new_points) return 0;
        
        space->points = new_points;
        space->capacity = new_capacity;
    }
    
    space->points[space->size] = *t;
    space->size++;
    return 1;
}

void tetralemma_space_print(const TetralemmaSpace* space) {
    if (!space) return;
    
    printf("Tetralemma Space (𝕋) - %d points:\n", space->size);
    for (int i = 0; i < space->size; i++) {
        printf("  Point %d: ", i);
        tetrapoint_print(&space->points[i]);
        printf("\n");
    }
}

// Emptiness as Limit: lim(n→∞) τⁿ(t) = Ψ
Tetrapoint tetralemma_space_emptiness_limit(const TetralemmaSpace* space, int iterations) {
    if (!space || space->size == 0) {
        Tetrapoint empty = {EMPTY, EMPTY, EMPTY, EMPTY};
        return empty;
    }
    
    Tetrapoint current = space->points[0];
    
    for (int i = 0; i < iterations; i++) {
        Tetrapoint next = tetrapoint_negation_transform(&current);
        
        // Check if we've reached emptiness
        if (tetrapoint_is_empty(&next)) {
            return next;
        }
        
        // Check if we've reached a cycle (same point)
        if (tetrapoint_equals(&current, &next)) {
            break;
        }
        
        current = next;
    }
    
    return current;
}

// Utility functions

void print_negation_cycle(const Tetrapoint* initial, int steps) {
    printf("🌀 Negation Cycle (τ transformation):\n");
    Tetrapoint current = *initial;
    
    for (int i = 0; i <= steps; i++) {
        printf("Step %d: ", i);
        tetrapoint_print(&current);
        
        // Add descriptive labels
        if (i == 0) printf(" => P (Affirmation)");
        else if (i == 1) printf(" => ¬(P ∨ ¬P) (Neither)");
        else if (i == 2) printf(" => P ∧ ¬P (Both)");
        else if (i == 3) printf(" => ¬P (Negation)");
        else printf(" => τ^%d(P)", i);
        
        printf("\n");
        
        if (i < steps) {
            current = tetrapoint_negation_transform(&current);
        }
    }
}

void demonstrate_contradiction_fusion(void) {
    printf("\n🔥 Contradiction Fusion (⊗ product):\n");
    
    Tetrapoint t1 = tetrapoint_create(EXPRESSED, SUPPRESSED, EXPRESSED, SUPPRESSED);
    Tetrapoint t2 = tetrapoint_create(EXPRESSED, EXPRESSED, SUPPRESSED, SUPPRESSED);
    
    printf("t1 = ");
    tetrapoint_print(&t1);
    printf("\n");
    
    printf("t2 = ");
    tetrapoint_print(&t2);
    printf("\n");
    
    Tetrapoint fusion = tetrapoint_contradiction_product(&t1, &t2);
    printf("t1 ⊗ t2 = ");
    tetrapoint_print(&fusion);
    printf(" (Contradiction produces collapse or intensification)\n");
} 
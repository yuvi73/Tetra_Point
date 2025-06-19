#ifndef TETRALEMMA_H
#define TETRALEMMA_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

// Base polarity values for Tetralemma Space
typedef enum {
    EXPRESSED = 1,    // 1: expressed
    SUPPRESSED = 0,   // 0: suppressed  
    INAPPLICABLE = -1, // Ø: inapplicable (represented as -1)
    EMPTY = -2        // Ψ: empty (represented as -2)
} Polarity;

// Tetrapoint structure: (a, ¬a, a∧¬a, ¬(a∨¬a))
typedef struct {
    Polarity a;           // Affirmation
    Polarity not_a;       // Negation
    Polarity both;        // Both true and false
    Polarity neither;     // Neither true nor false
} Tetrapoint;

// Tetralemma Space structure
typedef struct {
    Tetrapoint* points;
    int capacity;
    int size;
} TetralemmaSpace;

// Function declarations

// Tetrapoint operations
Tetrapoint tetrapoint_create(Polarity a, Polarity not_a, Polarity both, Polarity neither);
void tetrapoint_print(const Tetrapoint* t);
char* polarity_to_string(Polarity p);
Tetrapoint tetrapoint_negation_transform(const Tetrapoint* t);
Tetrapoint tetrapoint_contradiction_product(const Tetrapoint* t1, const Tetrapoint* t2);
int tetrapoint_is_empty(const Tetrapoint* t);
int tetrapoint_equals(const Tetrapoint* t1, const Tetrapoint* t2);

// Tetralemma Space operations
TetralemmaSpace* tetralemma_space_create(int initial_capacity);
void tetralemma_space_destroy(TetralemmaSpace* space);
int tetralemma_space_add(TetralemmaSpace* space, const Tetrapoint* t);
void tetralemma_space_print(const TetralemmaSpace* space);
Tetrapoint tetralemma_space_emptiness_limit(const TetralemmaSpace* space, int iterations);

// Utility functions
void print_negation_cycle(const Tetrapoint* initial, int steps);
void demonstrate_contradiction_fusion(void);

#endif // TETRALEMMA_H 
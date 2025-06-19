#include "tetralemma.h"

int main() {
    printf("🧠 Tetralemma Space (𝕋) - Mathematical Structure Implementation\n");
    printf("=============================================================\n\n");
    
    // Create a Tetralemma Space
    TetralemmaSpace* space = tetralemma_space_create(10);
    if (!space) {
        printf("Error: Failed to create Tetralemma Space\n");
        return 1;
    }
    
    // Create some example tetrapoints
    Tetrapoint p1 = tetrapoint_create(EXPRESSED, SUPPRESSED, SUPPRESSED, SUPPRESSED);  // P
    Tetrapoint p2 = tetrapoint_create(SUPPRESSED, EXPRESSED, SUPPRESSED, SUPPRESSED);  // ¬P
    Tetrapoint p3 = tetrapoint_create(SUPPRESSED, SUPPRESSED, EXPRESSED, SUPPRESSED);  // P ∧ ¬P
    Tetrapoint p4 = tetrapoint_create(SUPPRESSED, SUPPRESSED, SUPPRESSED, EXPRESSED);  // ¬(P ∨ ¬P)
    
    // Add points to the space
    tetralemma_space_add(space, &p1);
    tetralemma_space_add(space, &p2);
    tetralemma_space_add(space, &p3);
    tetralemma_space_add(space, &p4);
    
    // Print the space
    tetralemma_space_print(space);
    
    // Demonstrate negation cycle
    printf("\n");
    print_negation_cycle(&p1, 5);
    
    // Demonstrate contradiction fusion
    demonstrate_contradiction_fusion();
    
    // Test emptiness limit
    printf("\n🧘‍♂️ Emptiness as Limit Test:\n");
    Tetrapoint limit = tetralemma_space_emptiness_limit(space, 10);
    printf("Limit after 10 iterations: ");
    tetrapoint_print(&limit);
    printf("\n");
    
    // Test with different starting points
    printf("\n🔄 Testing different starting points:\n");
    Tetrapoint test_points[] = {
        tetrapoint_create(EXPRESSED, EXPRESSED, SUPPRESSED, SUPPRESSED),
        tetrapoint_create(INAPPLICABLE, SUPPRESSED, EXPRESSED, INAPPLICABLE),
        tetrapoint_create(EMPTY, EMPTY, EMPTY, EMPTY)
    };
    
    for (int i = 0; i < 3; i++) {
        printf("Starting point %d: ", i);
        tetrapoint_print(&test_points[i]);
        printf("\n");
        
        Tetrapoint result = tetrapoint_negation_transform(&test_points[i]);
        printf("After τ transform: ");
        tetrapoint_print(&result);
        printf("\n\n");
    }
    
    // Test contradiction product with various combinations
    printf("🔥 Advanced Contradiction Fusion Tests:\n");
    Tetrapoint complex1 = tetrapoint_create(EXPRESSED, INAPPLICABLE, SUPPRESSED, EMPTY);
    Tetrapoint complex2 = tetrapoint_create(INAPPLICABLE, EXPRESSED, INAPPLICABLE, SUPPRESSED);
    
    printf("Complex t1 = ");
    tetrapoint_print(&complex1);
    printf("\n");
    
    printf("Complex t2 = ");
    tetrapoint_print(&complex2);
    printf("\n");
    
    Tetrapoint complex_fusion = tetrapoint_contradiction_product(&complex1, &complex2);
    printf("Complex t1 ⊗ t2 = ");
    tetrapoint_print(&complex_fusion);
    printf("\n");
    
    // Test emptiness detection
    printf("\n🧘‍♂️ Emptiness Detection:\n");
    Tetrapoint empty_test = tetrapoint_create(EMPTY, EMPTY, EMPTY, EMPTY);
    printf("Empty tetrapoint: ");
    tetrapoint_print(&empty_test);
    printf(" is_empty: %s\n", tetrapoint_is_empty(&empty_test) ? "true" : "false");
    
    printf("Regular tetrapoint: ");
    tetrapoint_print(&p1);
    printf(" is_empty: %s\n", tetrapoint_is_empty(&p1) ? "true" : "false");
    
    // Clean up
    tetralemma_space_destroy(space);
    
    printf("\n✅ Tetralemma Space demonstration completed!\n");
    printf("This structure embodies the Catuṣkoṭi logic with:\n");
    printf("- Four-valued polarity system\n");
    printf("- Non-linear negation topology\n");
    printf("- Contradiction as generative relation\n");
    printf("- Emptiness as mathematical limit\n");
    
    return 0;
} 
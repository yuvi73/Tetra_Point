CC = gcc
CFLAGS = -Wall -Wextra -std=c99 -O2
TARGET = tetralemma
SOURCES = main.c tetralemma.c
OBJECTS = $(SOURCES:.c=.o)
HEADERS = tetralemma.h

.PHONY: all clean run

all: $(TARGET)

$(TARGET): $(OBJECTS)
	$(CC) $(OBJECTS) -o $(TARGET)

%.o: %.c $(HEADERS)
	$(CC) $(CFLAGS) -c $< -o $@

run: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(OBJECTS) $(TARGET)

# Development targets
debug: CFLAGS += -g -DDEBUG
debug: $(TARGET)

# Install dependencies (if needed)
install-deps:
	@echo "No external dependencies required for C implementation"

# Test compilation
test-compile: $(TARGET)
	@echo "✅ Compilation successful!"
	@echo "Run './tetralemma' to execute the program" 
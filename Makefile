CC = gcc
CFLAGS = -Wall -Wextra -O3
LDFLAGS = -lm
EXEC = main
SRC = src
OBJ = obj
SRC = src
INCLUDE = include
BIN = bin

all: $(EXEC)

$(EXEC): $(OBJ)/main.o $(OBJ)/function_matrix.o $(OBJ)/network.o
	$(CC) $(OBJ)/*.o -o $(BIN)/$(EXEC) $(LDFLAGS)

$(OBJ)/main.o: $(SRC)/main.c
	$(CC) $(CFLAGS) -I./$(INCLUDE) -c $(SRC)/main.c -o $(OBJ)/main.o

$(OBJ)/function_matrix.o: $(SRC)/function_matrix.c
	$(CC) $(CFLAGS) -I./$(INCLUDE) -c $(SRC)/function_matrix.c -o $(OBJ)/function_matrix.o

$(OBJ)/network.o: $(SRC)/network.c
	$(CC) $(CFLAGS) -I./$(INCLUDE) -c $(SRC)/network.c -o $(OBJ)/network.o


$(BIN):
	mkdir -p $(BIN)

$(OBJ):
	mkdir -p $(OBJ)

clean:
	rm -f $(OBJ)/*.o $(BIN)/*

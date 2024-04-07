
Tic Tac Toe with Artificial Neural Network (ANN) Player
This C++ project implements a Tic Tac Toe game where one of the players is controlled by an Artificial Neural Network (ANN). The ANN learns to play the game effectively through training on historical gameplay data.

Components and Functionalities

1. Neuron Class and Connection Struct
The Neuron class represents individual neurons in the ANN.
Each neuron has connections to neurons in the next layer, represented by the Connection struct, which stores connection weights and delta weights for backpropagation.

2. Neural Net Class
The NeuralNet class manages the layers of neurons, training the ANN, and making predictions.
It uses a vector of layers (vector<Layer>) where each layer is a vector of neurons (Layer is typedef for vector<neuron>).
Functionalities include feedforward, backpropagation, updating weights, getting results, and calculating error.

3. Tic Tac Toe Class
Represents the Tic Tac Toe game logic and player interactions.
Uses a vector to maintain the game board state and a character array for display.
Provides methods for displaying the board, making moves (by the AI or player), checking for wins or draws, and playing the game.

5. Main Function
Initializes the game and starts playing.
Uses the tictactoe class to initiate the game.
Includes a simple randomization function for generating random numbers to initialize the neural network weights.
6. Training
The train() method within the tictactoe class reads training data from a file (tic-tac-toe.txt) to train the ANN.
The ANN is trained to predict the outcome of each game state (win/lose/draw) based on the board configuration.
7. Player Interactions
Allows players to make moves alternately, with the AI making moves based on the trained ANN predictions.
The game continues until a win, draw, or loss condition is met.

How to Run
- Ensure you have a C++ compiler installed on your system.
- Clone or download the repository to your local machine.
- Compile the code using your preferred C++ compiler.
- Run the compiled executable file to start playing Tic Tac Toe against the ANN player.

Contributors
This project is maintained by Itba Malahat and Lailoma Noor

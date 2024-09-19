//board 
var blockSize = 25;
var rows = 20;
var cols = 20;
var board;
var context;

//snake head
var snakeX = blockSize * 5;
var snakeY = blockSize * 5; 

var velocityX = 0;
var velocityY = 0;

var snakeBody = [];

//second snake head
var snake2X = blockSize * 10;
var snake2Y = blockSize * 10;

var velocity2X = 0;
var velocity2Y = 0;

var snake2Body = [];

//food 
var foodX;
var foodY;

var gameover = false;
var reward1 = 100;
var reward2 = 100;


var socket = io(); // adding socket to connect python and javascript 


socket.on('connect', () => {
    console.log('Connected to server');
});


socket.on('action', function(data) {
    console.log('Velocities received from server:', data);

    // Directly update the velocities for snake 1
    velocityX = data.velocityX;
    velocityY = data.velocityY;

    // Directly update the velocities for snake 2
    velocity2X = data.velocity2X;
    velocity2Y = data.velocity2Y;
    
});

window.onload = function() {
    board = document.getElementById('board');
    board.height = rows * blockSize;
    board.width = cols * blockSize;
    context = board.getContext('2d'); // used for drawing on the canvas
    placesnake();
    placeFood();
    document.addEventListener('keydown', ChangeDirection);
    setInterval(update, 1000 / 10);
}


//sending game state to python backend 
function emitter(){
    let gameState = {
        snake1: { x: snakeX/500, y: snakeY/500, body: snakeBody.length },
        snake1dir: { x: velocityX, y: velocityY },
        
        snake2: { x: snake2X/500, y:snake2Y/500, body: snake2Body.length },
        snake2dir: { x: velocity2X, y: velocity2Y },

        food: { x: foodX/500, y: foodY/500 },
        reward1: reward1,
        reward2: reward2,
        gameover: gameover

    };
    
    socket.emit('game_state_before_action', gameState);
}

function emitter2(){
    let gameState = {
        snake1: { x: snakeX/500, y: snakeY/500, body: snakeBody.length },
        snake1dir: { x: velocityX, y: velocityY },
        
        snake2: { x: snake2X/500, y:snake2Y/500, body: snake2Body.length },
        snake2dir: { x: velocity2X, y: velocity2Y },

        food: { x: foodX/500, y: foodY/500 },
        reward1: reward1,
        reward2: reward2,
        gameover: gameover

    };
    
    socket.emit('game_state_after_action', gameState);
}

//to give reward to snake if gets closer to food
function calculateDistance(x1, y1, x2, y2) {
    return Math.sqrt(Math.pow(x1 - x2, 2) + Math.pow(y1 - y2, 2));
}

function update() {
    
    if (gameover) { resetGame(); return; }

    context.fillStyle = "black";
    context.fillRect(0, 0, board.width, board.height);

    emitter();

    let initialDistanceSnake1 = calculateDistance(snakeX, snakeY, foodX, foodY);
    let initialDistanceSnake2 = calculateDistance(snake2X, snake2Y, foodX, foodY);


    // Move snake 1
    snakeX += velocityX * blockSize;
    snakeY += velocityY * blockSize;

    // Move snake 2
    snake2X += velocity2X * blockSize;
    snake2Y += velocity2Y * blockSize;


    // Check for collision with food for snake 1
    if (snakeX == foodX && snakeY == foodY) {
        snakeBody.push([foodX, foodY]);
        reward1 = 1;
        document.getElementById('score1').innerText = reward1;
        placeFood();
    }

    // Check for collision with food for snake 2
    if (snake2X == foodX && snake2Y == foodY) {
        snake2Body.push([foodX, foodY]);
        reward2 = 1;
        document.getElementById('score2').innerText = reward2;
        placeFood();
    }
    //snake2
    if (snake2X != foodX && snake2Y != foodY) {
    
        // Calculate the new distance after the move
        let newDistanceSnake2 = calculateDistance(snake2X, snake2Y, foodX, foodY);
        
        // Reward Snake 2 if it gets closer to the food
        if (newDistanceSnake2 < initialDistanceSnake2) {
            reward2 = 0.05; // Increase this value based on the reward you want
        } else {
            reward2 = -0.03;  // Penalize if it moves away
        }
    }
    //snake 1 
    if (snakeX != foodX && snakeY != foodY) {
    
        // Calculate the new distance after the move
        let newDistanceSnake1 = calculateDistance(snakeX, snakeY, foodX, foodY);
        
        // Reward Snake 2 if it gets closer to the food
        if (newDistanceSnake1 < initialDistanceSnake1) {
            reward1 = 0.05; // Increase this value based on the reward you want
        } else {
            reward1 = -0.03;   // Penalize if it moves away
        }
    }
    

    // Update snake 1 body
    for (let i = snakeBody.length - 1; i > 0; i--) {
        snakeBody[i] = [...snakeBody[i - 1]];
    }
    if (snakeBody.length > 0) {
        snakeBody[0] = [snakeX - velocityX * blockSize, snakeY - velocityY * blockSize];
    }

    // Update snake 2 body
    for (let i = snake2Body.length - 1; i > 0; i--) {
        snake2Body[i] = [...snake2Body[i - 1]];
    }
    if (snake2Body.length > 0) {
        snake2Body[0] = [snake2X - velocity2X * blockSize, snake2Y - velocity2Y * blockSize];
    }

    
     // Draw snake 1 with eyes
    drawSnake(snakeX, snakeY, snakeBody, 'lime');
    
    // Draw snake 2 with eyes
    drawSnake(snake2X, snake2Y, snake2Body, 'blue');

    // Draw food as an apple
    drawApple(foodX, foodY);

    // Draw food
    // context.fillStyle = "red";
    // context.fillRect(foodX, foodY, blockSize, blockSize);

    
    // Check for game over conditions for snake 1
    if (snakeX < 0 || snakeX >= cols * blockSize || snakeY < 0 || snakeY >= rows * blockSize) {
        reward1 = -1;
        gameover = true;
        
 
        //alert('Game Over! Snake 1 Reward: ' + reward1 + ', Snake 2 Reward: ' + reward2);
        
    }

    for (let i = 0; i < snakeBody.length; i++) {
        if (snakeX == snakeBody[i][0] && snakeY == snakeBody[i][1]) {
            reward1 = -1;
            gameover = true;
            

            //alert('Game Over! Snake 1 Reward: ' + reward1 + ', Snake 2 Reward: ' + reward2);
            return;
        }
    }

    // Check for game over conditions for snake 2
    if (snake2X < 0 || snake2X >= cols * blockSize || snake2Y < 0 || snake2Y >= rows * blockSize) {
        reward2 =-1;
        gameover = true;
        

        //alert('Game Over! Snake 1 Reward: ' + reward1 + ', Snake 2 Reward: ' + reward2);
        return;
    }

    for (let i = 0; i < snake2Body.length; i++) {
        if (snake2X == snake2Body[i][0] && snake2Y == snake2Body[i][1]) {
            reward2 = -1;
            gameover = true;
            

            //alert('Game Over! Snake 1 Reward: ' + reward1 + ', Snake 2 Reward: ' + reward2);

            return;
        }
    }

    // Check for collision between the two snakes
    for (let i = 0; i < snakeBody.length; i++) {
        if (snake2X == snakeBody[i][0] && snake2Y == snakeBody[i][1]) {
            reward2 = -1;
            gameover = true;
           
  
            //alert('Game Over! Snake 1 Reward: ' + reward1 + ', Snake 2 Reward: ' + reward2);
    
            return;
        }
    }

    for (let i = 0; i < snake2Body.length; i++) {
        if (snakeX == snake2Body[i][0] && snakeY == snake2Body[i][1]) {
            reward1 = -1;
            gameover = true;
           
      
            //alert('Game Over! Snake 1 Reward: ' + reward1 + ', Snake 2 Reward: ' + reward2);
        
            return;
        }
    }

    // Check for head-to-head collision
    if (snakeX == snake2X && snakeY == snake2Y) {
        reward1 = -1;
        reward2 = -1;
        gameover = true;
        
        //alert('Game Over! Snake 1 Reward: ' + reward1 + ', Snake 2 Reward: ' + reward2);

        return;
    }
    
    emitter2();
    
}

function ChangeDirection(event) {
    // Control snake 1 with arrow keys
    if (event.keyCode == 38 && velocityY != 1) { //up
        velocityY = -1;
        velocityX = 0;
    }
    else if (event.keyCode == 40 && velocityY != -1) { //down
        velocityY = 1;
        velocityX = 0;
    }
    else if (event.keyCode == 37 && velocityX != 1) { //left
        velocityX = -1;
        velocityY = 0;
    }
    else if (event.keyCode == 39 && velocityX != -1) { //right
        velocityX = 1;
        velocityY = 0;
    }

    // Control snake 2 with W, A, S, D
    if (event.keyCode == 87 && velocity2Y != 1) { // W key - up
        velocity2Y = -1;
        velocity2X = 0;
    }
    else if (event.keyCode == 83 && velocity2Y != -1) { // S key - down
        velocity2Y = 1;
        velocity2X = 0;
    }
    else if (event.keyCode == 65 && velocity2X != 1) { // A key - left
        velocity2X = -1;
        velocity2Y = 0;
    }
    else if (event.keyCode == 68 && velocity2X != -1) { // D key - right
        velocity2X = 1;
        velocity2Y = 0;
    }
}

function placeFood() {
    var validPosition = false;
    while (!validPosition) {
        foodX = Math.floor(Math.random() * cols) * blockSize;
        foodY = Math.floor(Math.random() * rows) * blockSize;

        // Check if the food spawns on snake 1
        if ((snakeX === foodX && snakeY === foodY) ||
            snakeBody.some(segment => segment[0] === foodX && segment[1] === foodY)) {
            continue;
        }

        // Check if the food spawns on snake 2
        if ((snake2X === foodX && snake2Y === foodY) ||
            snake2Body.some(segment => segment[0] === foodX && segment[1] === foodY)) {
            continue;
        }

        validPosition = true;
    }
}

function placesnake() {
    let validPosition = false;

    while (!validPosition) {
        // Place snake 1
        snakeX = Math.floor(Math.random() * cols) * blockSize;
        snakeY = Math.floor(Math.random() * rows) * blockSize;

        // Place snake 2
        let tempSnake2X = Math.floor(Math.random() * cols) * blockSize;
        let tempSnake2Y = Math.floor(Math.random() * rows) * blockSize;

        // Check if snake 2's position overlaps with snake 1
        if (tempSnake2X !== snakeX || tempSnake2Y !== snakeY) {
            snake2X = tempSnake2X;
            snake2Y = tempSnake2Y;
            validPosition = true;
        }
    }
}

function resetGame() {
    // snakeX = blockSize * 5;
    // snakeY = blockSize * 5;
    reward1=0;
    reward2=0;
    velocityX = 0;
    velocityY = 0;
    snakeBody = [];

    // snake2X = blockSize * 10;
    // snake2Y = blockSize * 10;
    velocity2X = 0;
    velocity2Y = 0;
    snake2Body = [];
    placesnake()
    gameover = false;
    placeFood();
}

// Function to draw the snake with eyes
function drawSnake(x, y, body, color) {
    context.fillStyle = color;
    context.fillRect(x, y, blockSize, blockSize);

    // Draw eyes
    context.fillStyle = 'white';
    context.beginPath();
    context.arc(x + blockSize / 4, y + blockSize / 4, blockSize / 8, 0, Math.PI * 2);
    context.fill();

    context.beginPath();
    context.arc(x + 3 * blockSize / 4, y + blockSize / 4, blockSize / 8, 0, Math.PI * 2);
    context.fill();

    // Draw the rest of the snake body
    context.fillStyle = color;
    for (let i = 0; i < body.length; i++) {
        context.fillRect(body[i][0], body[i][1], blockSize, blockSize);
    }
}

// Function to draw the apple food
function drawApple(x, y) {
    context.fillStyle = 'red';
    context.beginPath();
    context.arc(x + blockSize / 2, y + blockSize / 2, blockSize / 2, 0, Math.PI * 2);
    context.fill();

    // Draw apple stem
    context.fillStyle = 'brown';
    context.beginPath();
    context.moveTo(x + blockSize / 2, y + blockSize / 2 - blockSize / 2);
    context.lineTo(x + blockSize / 2 - 5, y + blockSize / 2 - blockSize / 2 - 10);
    context.lineTo(x + blockSize / 2 + 5, y + blockSize / 2 - blockSize / 2 - 10);
    context.closePath();
    context.fill();
}

function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%

    % Step 1: Recode y from a number to a bit pattern
    e = eye(num_labels); 
    yy = e(y,:);                    % [5000x10]

    % Step 2: page 4
    a1 = [ones( m,1), X];           % [5000x401]

    % hidden layer
    z2 = Theta1 * a1';              % [25x401]x[401*5000]=[25x5000]
    a2 = sigmoid( z2);              % [25x5000]
    a2 = [ones(1, size(a2,2)); a2]; % [26 x 5,000]

    z3 = Theta2 * a2;               % [10x26]x[26x5000]=[10x5000]
    h  = sigmoid( z3);              % [10x5000]

    jj = -yy .* log(h)' - (1 - yy) .* log(1 - h)';
    J = (1/m) * sum(jj(:));

    % regularization
    Theta1_to_2 = Theta1(:,2:end) .^2;
    Theta2_to_2 = Theta2(:,2:end) .^2;
    SumTheta1 = [ sum( Theta1_to_2(:)) ; sum( Theta2_to_2(:))];
    reg = lambda / (2 * m) * sum( SumTheta1);
    J = J + reg;

% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
    delta1 = 0;
    delta2 = 0;

    for t=1:m
        
        % Step 1 - Feed forward
        a1 = [ 1; X(t,:)'];                 % [401x1]
        z2 = Theta1 * a1;                   % [25 x 1] = [25x401]x[401x1]
        a2 = sigmoid(z2);
        a2 = [ones(1, size(a2,2)); a2];      % [26 x 1] 
        z3 = Theta2 * a2;                   % [10x1] = [10x26]x[26x1]
        a3 = sigmoid( z3);                  % [10x1]
        
        % Step 2 - Subtract results from output
        d3 = a3 - e(:,y(t,:));              % [10x1]
        
        % Step 3 - Back propagate error
        d2 = (Theta2(:,2:end)' * d3) .* sigmoidGradient( z2); % [25x1]
        
        % Step 4 - accumulate errors
        delta1 = delta1 + d2 * a1';         % [25x401]
        delta2 = delta2 + d3 * a2';         % [10x26]
        
    end
    
    Theta1_grad = delta1 / m;
    Theta2_grad = delta2 / m;
    
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

Theta1_grad(:,2:end) = Theta1_grad(:,2:end) + (lambda/m) * Theta1(:,2:end);
Theta2_grad(:,2:end) = Theta2_grad(:,2:end) + (lambda/m) * Theta2(:,2:end);

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end

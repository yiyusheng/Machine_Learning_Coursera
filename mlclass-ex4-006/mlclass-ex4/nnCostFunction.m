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
kind_y=length(unique(y));
A2=sigmoid(Theta1*[ones(size(X,1),1),X]');
A3=sigmoid(Theta2*[ones(1,size(A2,2));A2]);
for i=1:m
	Y=zeros(kind_y,1);
	if y(i,1)==0
		Y(end,1)=1;
	else
		Y(y(i,1),1)=1;
	end
	J=J+(1/m)*((-Y'*log(A3(:,i)))-(1-Y')*log(1-A3(:,i)));
end
%regularized nnCostFunction
T1=Theta1(:,[2:end]);
T2=Theta2(:,[2:end]);
J=J+(lambda/(2*m))*(T1(:)'*T1(:)+T2(:)'*T2(:));
%
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
%START FROM HERE
kind_y=length(unique(y));
Z2=Theta1(:,[2:end])*X';
A2=sigmoid(Theta1*[ones(size(X,1),1),X]');
Z3=Theta2(:,[2:end])*A2;
A3=sigmoid(Theta2*[ones(1,size(A2,2));A2]);
for i=1:m
	%layer3
	Y=zeros(kind_y,1);
	if y(i,1)==0
		Y(end,1)=1;
	else
		Y(y(i,1),1)=1;
	end
	delta3=A3(:,i)-Y;
	%layer2
	delta2=(theta2'*delta3).*sigmoidGredient(Z2);
	delta2(:,[2:end])=delta2(:,[2:end])+delta3*A2(:,i)'
end
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%



















% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end

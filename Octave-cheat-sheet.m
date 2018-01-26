%apt-get install octave


A = [1, 2, 3; 4, 5, 6; 7, 8, 9; 10, 11, 12]     % The ; denotes we are going back to a new row.
v = [1;2;3]                                     % Initialize a vector  
[m,n] = size(A)                                 % Get the dimension of the matrix A where m = rows and n = columns
dim_A = size(A)                                 % You could also store it this way
dim_v = size(v)                                 % Get the dimension of the vector v 
A_23 = A(2,3)                                   % Now let's index into the 2nd row 3rd column of matrix A

% Initialize matrix A and B 
A = [1, 2, 4; 5, 3, 2]
B = [1, 3, 4; 1, 1, 1]
s = 2                                           % Initialize constant s 
add_AB = A + B                                  % See how element-wise addition works
sub_AB = A - B                                  % See how element-wise subtraction works
mult_As = A * s                                 % See how scalar multiplication works
div_As = A / s                                  % Divide A by s
add_As = A + s                                  % What happens if we have a Matrix + scalar?


A = [1, 2, 3; 4, 5, 6;7, 8, 9]                  % Initialize matrix A 
v = [1; 1; 1]                                   % Initialize vector v 
Av = A * v                                      % Multiply A * v


A = [1, 2; 3, 4;5, 6]                           % Initialize a 3 by 2 matrix 
B = [1; 2]                                      % Initialize a 2 by 1 matrix 
mult_AB = A*B                                   % We expect a resulting matrix of (3 by 2)*(2 by 1) = (3 by 1) 


% Initialize random matrices A and B 
A = [1,2;4,5]
B = [1,1;0,2]
I = eye(2)                                      % Initialize a 2 by 2 identity matrix; notation is the same as I = [1,0;0,1]
IA = I*A                                        % What happens when we multiply I*A ? 
AI = A*I                                        % How about A*I ? 
AB = A*B                                        % Compute A*B 
BA = B*A                                        % Is it equal to B*A? 
% Note that IA = AI but AB != BA


% Initialize matrix A 
A = [1,2,0;0,5,6;7,0,9]
A_trans = A'                                    % Transpose A 
A_inv = inv(A)                                  % Take the inverse of A 
A_invA = inv(A)*A                               % What is A^(-1)*A? 

pinv(X'*X)*X'*y

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


1 == 2
1 ~= 2
1 && 0
1 || 0
xor(1,0)





%%% BASIC COMMANDS
% Some Linux cmd like: ls, cd, pwd
PS1('>> ');      % Change prompt to >>      % semicolon supress output print

addpath('D:\Google Drive\EDUC\DataScience\scripts\Octave')   % Add a search PATH
help [command]

disp(a);     % print a
disp(sprintf('2 decimals: %0.2f', a))     % 

load file.ext            % load files
load ('file.ext')
save var1.mat X          % save X to file var1.mat
save var1.txt X -ascii   % save X to file var1.mat

who                      % show variables in the workspace
whos                     % show variables with details (name, size, class)
clear                    % clear all variables 
clear X                  % clear VariableX from workspace
X(1:10)                  % show rows 1 to 10 (Slice)


%%% MATRIX and VECTOR 

vector = 1:0.1:2       % from 1 to 2 incrementing by 0.1

eye(4)                 % Identity Matrix 4x4  (1s in diagonal)
ones(2,3)              % create a 2x3 matrix filled with 1
zeros(2,3)             % create a 2x3 matrix filled with 0
rand(2,3)              % 2x3 matrix with random numbers
rand(2,3)              % 2x3 matrix with random numbers in Gaussian Distribuition (média 0, variancia ou std 1)
size(A)                % display size of matrix A
size(A,1)              % display the ROW size of matrix A
size(A,2)              % display the COLUMN size of matrix A
lenght(A)              % Longer dimension of A  = len(A)
magic(3)               % Magic Square



A(3,2)                 % Show element in row 3, column 3
A(2,:)                 % All elements in row 2
A(:,1)                 % All elements in column 1
A([1 3],:)             % All elements in row 1 and 3
A(:,2) = [10;11;12]    % Assign elements to column 2
A = [A, [10;20;30]]    % append new column
A(:)                   % All elements of A into a single vector 

C = [A B]              % concatenate matrix A and B side by side
C = [A; B]             % concatenate matrix A and B on top

%%% COMPUTING DATA

A*C                     % multiply Matrix
A.*B                    % multiply elements on matrix         . indicates element-wise operations
A.^2                    % square of elements in A
1./A                    % 1 over elements
log(A)                  %
exp(A)
abs(A)
A'                      % Transpose A
max(A,[],1)             % maximum elements in the Column
max(A,[],2)             % maximum elements in the Row
max(max(A))             % maximum element in the matrix        = max(A(:))
sum(A,1)                % sum per columns
sum(A,2)                % sum per rows
sum(sum(A.*eye(9)))     % sum diagonal in 9x9 matrix
pinv(A)                 % Inverse Matrix

val = max(v)            % maximum value of A 
[val, ind] = max(v)     % value and indice of maximum elements in A
v < 3                   % return 1 or 0 for elements in vector
find(v < 3)             % return elements in condition

sum(v)
prod(v)
floor(v)                % round down to integer
ceil(v)                 % round up to integer

%%% PLOT GRAPH

plot(x,y, 'r')         % Line in red
hold on;               % keep prev graph so you can plot another above
xlabel('')
ylabel('')
legend('')
title('')
axis(xmin xmax ymin ymax])
print -dpng 'graph1.png'
close;                 % close plot, disable hold on

figure(1); plot(x1,y1);
figure(2); plot(x2,y2);

subplot(1,2,1); plot(x1,y1);
subplot(1,2,2); plot(x2,y2);

hist(x, 20)            % histogram of x, with 20 bins
imagesc(A)             % plot matrix with diff colors
imagesc(A), colorbar,  colormap gray  % plot matrix with color bar in the side, using grayscale
plot(x, y, 'rx', 'MarkerSize', 10);   % Scatter with red markers x

%%%%%%%%%%% COMMANDS

for i=1:10,
   disp(i);
end;

while i <=5,
   v(i) =100
   i=i+1;
end;

if i > 6,
   disp('Maior');
elseif i < 6,
   disp('Menor');
else
   disp('Igual');
end;


%%%%% FUNCTIONS

ex: % Will be included in distinct file

function y = squareThisNumber(x)     % will return y, and receive x
y = x^2;

function [y1, y2] = squareAndCubeThisNumber(x)    % return y1 and y2
y1 = x^2;
y2 = x^3;




%Neural Networks - Srinivasan Rengarajan- 50097996%
function [w1 w2]= nn_train(X_data)
warning('off');
%load X_data.mat;
d= X_data(:,1);
X_data = X_data(:,2:end);
M=20; % Corresponds to the number of hidden neurons%
m=size(X_data);
error=0; % Initial value of Error%
X_data = [ones(m,1) X_data];
% Setting the target matrix to achieve 1 of K Classification mechanism%
for k=1:19978
   if d(k,1)==0
    target=[1 0 0 0 0 0 0 0 0 0];
    tn(k,:) = target;  
end
if d(k,1 )==1
   target=[0 1 0 0 0 0 0 0 0 0];
     tn(k,:)= target;    
end
if d(k,1)==2
    target=[0 0 1 0 0 0 0 0 0 0];
    tn(k,:)= target;   
 end
 if d(k,1)==3
     target=[0 0 0 1 0 0 0 0 0 0];   
     tn(k,:)= target;    
 end
 if d(k,1)==4
     target=[0 0 0 0 1 0 0 0 0 0];   
     tn(k,:)= target;    
 end
 if d(k,1)==5
     target=[0 0 0 0 0 1 0 0 0 0];               
     tn(k,:)= target;    
 end
 if d(k,1)==6
     target=[0 0 0 0 0 0 1 0 0 0];
     tn(k,:)= target;    
 end
 if d(k,1)==7
     target=[0 0 0 0 0 0 0 1 0 0];  
     tn(k,:)= target;    
 end
     
     if d(k,1)==8
     target=[0 0 0 0 0 0 0 0 1 0];     
     tn(k,:)= target;    
     end
     
     if d(k,1)==9
     target=[0 0 0 0 0 0 0 0 0 1];       
     tn(k,:)= target;    
    end
end
% Randomize the value of w1 and w2 in order to vary from [-0.5 0.5]
w1= rand(513,M)-0.5;
w2= rand(10,M)-0.5;
% Minimize the Error for 10000 Iterations and generate the output%
for c = 1:300
disp(c);
%Activation function%
aj= X_data*w1;
Zj=tanh(aj);
yk= Zj*w2';
for i=1:19978
        for j=1:10
            yk(i,j) = exp(yk(i,j))/sum(exp(yk(i,:)));
        end
end
%Backpropagation -error computation%
%I= ones(19978,M);
delta_k = yk-tn;
delta_j = (1- (Zj.^2)).* (delta_k *w2);
eg1 =(X_data)'*delta_j;
eg2= (delta_k)'*Zj;
l=0.0001; % value of learning rate%
w1_new = w1 - l*eg1;
w2_new = w2 - l*eg2;
w1= w1_new;
w2= w2_new;
for i=1:19978
    for j=1:10
       error = error+(tn(i,j)*log(yk(i,j)));
       error=error*-1;
 end
end
E(c)= error;
error=0;
end
end






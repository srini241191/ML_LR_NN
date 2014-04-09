function [wnew]= train_lr(X_data)
warning('off');
d= X_data(:,1);
X_data = X_data(:,2:end);
m=size(X_data);
X_data= [ones(m,1) X_data];
%Create random matrix of w%
target=[0 0 0 0 0 0 0 0 0 0];
w=rand(513,10);
tn=zeros(19978,10);
E=zeros(5000,1);
for k=1:19978
   if d(k,1)==0
    target=[1 0 0 0 0 0 0 0 0 0];
    tn(k,:)= target;   
   end
 if d(k,1)==1
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
for c=1:5000
yn= X_data*w;
[m n]= size(yn);
for i=1:m
    for j=1:n
        yn(i,j) = exp(yn(i,j))/sum(exp(yn(i,:)));
        end
end
%yn=yn';
[m n]= size(yn);
eg =X_data'*(yn-tn);
alpha =0.001;
wnew = w - alpha*eg;
w=wnew;
Error=0;
for i=1:19978
    for j=1:10
        Error= Error+tn(i,j)*log(yn(i,j));
    end
end
E(c)=Error;
Error=0;
end
end


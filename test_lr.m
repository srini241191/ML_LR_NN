function [y predict hits r_rank]= test_lr(wnew, test_data)
%The label matrix corresponds to values from 0-9 written for the Train and
%test Data, considered as the Ground Truth%
labels= test_data(:,1);
test_data = test_data(:,2:end);
m=size(test_data);
test_data= [ones(m,1) test_data];
%To find the maximum output of the classifier
yk= test_data*wnew;
%Built -in function to predict the  maximum output of the classfier%
[y predict]= max(yk,[],2);
predict = predict-1;
%To find the accuracy of the prediction- the number of hits ie, the number
%of correctly classified numbers from 0-9 is calculated by estimating the
%number of properly classified instances in comparison to the label values%
for i=1:1500
if (labels(i,1) == (predict(i,1)))
    hits(i,1) =1;
end
end
Accuracy= sum(hits)/m(1)*100;
fprintf('The Accuracy of prediction for Logistic Regression  is');
disp(Accuracy);
Error_test = 1- Accuracy/100;
fprintf('The Test Error rate  for Logistic Regression is');
disp(Error_test*100);
%The reciprocal rank is defined as the computation of rank as compared to
%the first occurance of the relevant class label%
for i=1:1500
if(predict(1)==0)
    r_rank=1;
else
    index = find(predict(i)==0);
    r_rank = 1/index;
end
end
fprintf('The Reciprocal rank for Logistic Regression is');
disp(r_rank);
end




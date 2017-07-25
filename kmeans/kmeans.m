%select the input file
[fileName1,pathName1] = uigetfile('*.txt','Select the training data file');
trainData = csvread(strcat(pathName1,fileName1),1,0);
classData = csvread(strcat(pathName1,fileName1),0,0,[0, 0, 0, size(trainData,2)-1]);
%trainData = csvread('C:\Users\Roshan\Desktop\Fall 2016\Data Mining\Project3\Hand-Written-26-letters.txt',0,0);
%classData = csvread('C:\Users\Roshan\Desktop\Fall 2016\Data Mining\Project3\Hand-Written-26-letters.txt',0,0,[0, 0, 0, size(trainData,2)-1]);

% input value of K for k-means
prompt = 'enter value of k for k means ';
k=input(prompt);

classSize = size(classData,2);

%call kmeans implementation in matlab
kIndex= kmeans(trainData',k);
n=length(kIndex);

%creating the confusion matrix
conMat=zeros(k);
for i=1:n
        k2=kIndex(i);
        k1=classData(i);
        conMat(k1,k2)= conMat(k1,k2)+1;
end
disp(conMat)
%conMat
fprintf('Accuracy before reordering the confusion matrix:\n');
disp((sum(diag(conMat(:,:)))/classSize)*100);
%call munkres algorithm for linear assignment problems
[colInd] = munkres(-(conMat));
conMat(:,colInd)

%calculate accuracy
fprintf('Accuracy after reordering the confusion matrix:\n');
disp((sum(diag(conMat(:,colInd)))/classSize)*100);

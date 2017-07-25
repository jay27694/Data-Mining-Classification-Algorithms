import numpy as np
import csv
from numpy.linalg import inv

print("1: ATNT50, 2: ATNT200")
input_file = int(input("Enter dataset number: "))
if(input_file == 1):
    file_train = open("Data\\ATNT50\\csv\\trainDataXY.csv")
    file_test = open("Data\\ATNT50\\csv\\testDataXY.csv")
if(input_file == 2):
    file_train = open("Data\\ATNT200\\csv\\trainDataXY.csv")
    file_test = open("Data\\ATNT200\\csv\\testDataXY.csv")

file_reader_train = csv.reader(file_train)
Data_train = list(file_reader_train)

file_reader_test = csv.reader(file_test)
Data_test = list(file_reader_test)

def ConvertData(Data):
    #transpose
    New_Data = list(map(list, zip(*Data)))
    #extract class
    cls=[]
    for i in range(len(New_Data)):
        cls.append(New_Data[i].pop(0))
    #append class at last column
    for i, row in enumerate(New_Data):
        row.append(cls[i])

    for i in range(len(New_Data)):
        for j in range(len(New_Data[i])):
            New_Data[i][j] = float(New_Data[i][j])
    return New_Data


New_Train_Data = ConvertData(Data_train)
New_Test_Data = ConvertData(Data_test)

#extract train_class
Train_class = []
no = len(New_Train_Data[0])-1
for i in range(len(New_Train_Data)):
        Train_class.append(New_Train_Data[i].pop(no))

#extract test_class
Test_class = []
no = len(New_Test_Data[0])-1
for i in range(len(New_Test_Data)):
        Test_class.append(New_Test_Data[i].pop(no))

#get unique class
unique_cls = sorted(set(Train_class))

#Y_Train [no_of_ex*unique_cls]
temp=[]
Y_Train=[]
for i in range(len(unique_cls)):
    for j in range(len(Train_class)):
        if(Train_class[j]==unique_cls[i]):
            temp.append(1.0)
        else:
            temp.append(0.0)
    Y_Train.append(temp)
    temp=[]
Y_Train = np.array(Y_Train)
Y_Train = np.transpose(Y_Train)

#create X_Teat data [features*no_of_ex]
X_Test = np.array(New_Test_Data)
X_Test = np.transpose(X_Test)

#A-inv = (A')*(A*A')^(-1)
#create X_Train_inverse [features*no_of_ex]
X_Train = np.array(New_Train_Data)
X_Train_Transpose = np.transpose(New_Train_Data)
MUL = np.matmul(X_Train,X_Train_Transpose)
X = inv(MUL)
X_Train_Inverse = np.matmul(X_Train_Transpose,X)

#B = A^(-1)*Y
#B[features*unique_class] = X_Train_Inverse[features*no_of_ex] * Y_Train[no_of_ex*unique_cls]
B = np.matmul(X_Train_Inverse,Y_Train)
#B = B'[unique_class*features]
B = np.transpose(B)

#print(B)
#Y_Test = B' * X_Test
#Y_Test_Derived[unique_class*no_of_ex] = B[unique_class*features]* X_Test[features*no_of_ex]
Y_Test_Derived = np.matmul(B,X_Test)
#transpose of Y_Test_derived
Y_Test_Derived = np.transpose(Y_Test_Derived)
#get index of maximum value from each row (for each example)
#print(Y_Test_Derived)
x = Y_Test_Derived.argmax(axis=1)
Derived_class = x+1

print("=============================================================================================================")
print("Derived class with Linear Regression")
print(Derived_class)
print("=============================================================================================================")
print("Actual class")
actual_class=[]
counter = 0
for row in range(len(New_Test_Data)):
    actual_class.append(Test_class[row])
    if (Test_class[row] == Derived_class[row]):
        counter += 1
print(actual_class)
print("=============================================================================================================")
print("Accuracy in %")
print(counter*100/len(New_Test_Data))
print("=============================================================================================================")
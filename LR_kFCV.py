import numpy as np
import csv
from numpy.linalg import inv

#take datasets
print("1: ATNT50, 2: ATNT200, 3: Exam, 4: ATNT400, 5: Gene, 6: Hand ")
input_file = int(input("Enter dataset number: "))
if(input_file == 1):
    file_train = open("Data\\ATNT50\\csv\\trainDataXY.csv")
if(input_file == 2):
    file_train = open("Data\\ATNT200\\csv\\trainDataXY.csv")
if(input_file == 3):
    file_train = open("Data\\exam\\MtrainDataXY.csv")
if(input_file == 4):
    file_train = open("Data\\ATNT400\\ATNT400.csv")
if(input_file == 5):
    file_train = open("Data\\Gene\\Gene.csv")
if(input_file == 6):
    file_train = open("Data\\Hand\\Hand.csv")

file_reader_train = csv.reader(file_train)
Data_train = list(file_reader_train)


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

#convert data
New_Train_Data = ConvertData(Data_train)

#performing random number
import random
rannums = [x for x in range(len(New_Train_Data))]
random.shuffle(rannums)
#print(rannums)
eachFoldSize = int(len(New_Train_Data)/5)
#print(eachFoldSize)
temprow = []
temp = []
last = 0
for j in range(5):
    for i in range(eachFoldSize):
        temprow.append(rannums[last])
        last+=1
    temp.append(temprow)
    temprow=[]
#print(temp)


def methodcall(Train, Test):

    #extract train_class
    Train_class = []
    no = len(Train[0])-1
    for i in range(len(Train)):
            Train_class.append(Train[i].pop(no))

    #extract test_class
    Test_class = []
    no = len(Test[0])-1
    for i in range(len(Test)):
            Test_class.append(Test[i].pop(no))

    #get unique class
    unique_cls = sorted(set(Train_class))
    #print(unique_cls)

    #Y_Train [no_of_ex*unique_cls]
    temp1=[]
    Y_Train=[]
    for i in range(len(unique_cls)):
        for j in range(len(Train_class)):
            if(Train_class[j]==unique_cls[i]):
                temp1.append(1.0)
            else:
                temp1.append(0.0)
        Y_Train.append(temp1)
        temp1=[]
    Y_Train = np.array(Y_Train)
    Y_Train = np.transpose(Y_Train)

    #create X_Teat data [features*no_of_ex]
    X_Test = np.array(Test)
    X_Test = np.transpose(X_Test)

    #A-inv = (A')*(A*A')^(-1)
    #create X_Train_inverse [features*no_of_ex]
    X_Train = np.array(Train)
    X_Train_Transpose = np.transpose(Train)
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


    actual_class=[]
    counter = 0
    for row in range(len(Test)):
        actual_class.append(Test_class[row])
        if (Test_class[row] == Derived_class[row]):
            counter += 1

    return (counter*100/len(Test))

Train = []
Test = []
Answer = []
for i in range(len(temp)):
    for row in range(len(New_Train_Data)):
        if row  in temp[i]:
            Test.append(list(New_Train_Data[row]))
        else:
            Train.append(list(New_Train_Data[row]))
    Answer.append(methodcall(Train, Test))
    print(np.shape(Train))
    print(np.shape(Test))
    Test=[]
    Train=[]

print(Answer)
print(np.mean(Answer))
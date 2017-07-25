from sklearn import svm
import csv, numpy as np

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

    # extract classes from train
    Train_class = []
    no = len(Train[0]) - 1
    for i in range(len(Train)):
        Train_class.append(Train[i].pop(no))

    # extract classes from test
    Test_class = []
    no = len(Test[0]) - 1
    for i in range(len(Test)):
        Test_class.append(Test[i].pop(no))

    #run svm classifier
    clf = svm.SVC(kernel="linear")
    clf.fit(Train, Train_class)
    clf.decision_function_shape="ovo"

    #alpha value in SVC
    alphas = np.abs(clf.dual_coef_)
    #print(alphas)
    Derived_class = clf.predict(Test)


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
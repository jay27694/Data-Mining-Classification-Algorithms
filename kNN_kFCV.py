import csv, math
import operator
import itertools
import numpy as np

#take datasets
print("1: ATNT50, 2: ATNT200, 3: Exam, 4: ATNT400, 5: Gene, 6: Hand, 7: NBA")
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
if(input_file == 7):
    file_train = open("Data\\NBA\\new.csv")
file_reader_train = csv.reader(file_train)
Data_train = list(file_reader_train)

kNN_k = int(input("Enter k value: "))

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

def euclidean_dis(x, y):
    dist = 0.0
    for i in range(len(x) - 1):
        dist += pow((x[i] - y[i]), 2)
    dist = math.sqrt(dist)
    return dist

# get most common class from the list
def most_common(L):
    SL = sorted((x, i) for i, x in enumerate(L))
    groups = itertools.groupby(SL, key=operator.itemgetter(0))

    def _auxfun(g):
        item, iterable = g
        count = 0
        min_index = len(L)
        for _, where in iterable:
            count += 1
            min_index = min(min_index, where)
        return count, -min_index

    return max(groups, key=_auxfun)[0]


def methodcall(Train, Test):
    # calculate euclidean distance matrix
    Euclidean_Matrix_Row = []
    Euclidean_Matrix = []
    for i in range(len(Test)):
        for j in range(len(Train)):
            Euclidean_Matrix_Row.append(euclidean_dis(Test[i], Train[j]))
        Euclidean_Matrix.append(Euclidean_Matrix_Row)
        Euclidean_Matrix_Row = []

    find_repeat_list = []
    derived_class = []

    for row in range(len(Euclidean_Matrix)):
        # find the index of the euclidien distance in sorted order
        temp = sorted(range(len(Euclidean_Matrix[row])), key=lambda i: Euclidean_Matrix[row][i])
        # find kth nearest distance class
        for k in range(kNN_k):
            find_repeat_list.append(Train[temp[k]][len(Train[0]) - 1])
        # find common class from them
        com_cls = most_common(find_repeat_list)
        derived_class.append(com_cls)
        find_repeat_list = []

    actual_class = []
    counter = 0
    for row in range(len(Test)):
        actual_class.append(Test[row][len(Test[0]) - 1])
        if (Test[row][len(Test[0]) - 1] == derived_class[row]):
            counter += 1

    return (counter * 100 / len(Test))

TestRow = []
TrainRow = []
Train = []
Test = []
Answer = []
for i in range(len(temp)):
    for row in range(len(New_Train_Data)):
        if row  in temp[i]:
            Test.append(New_Train_Data[row])
            TestRow.append(1)
            TrainRow.append(0)
        else:
            Train.append(New_Train_Data[row])
            TestRow.append(0)
            TrainRow.append(1)
    #print(len(TrainRow))
    #print(len(TestRow))
    TrainRow=[]
    TestRow=[]
    Answer.append(methodcall(Train, Test))
    print(np.shape(Train))
    print(np.shape(Test))
    Test=[]
    Train=[]

print(Answer)
print(np.mean(Answer))


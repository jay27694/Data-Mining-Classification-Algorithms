import csv, math
import operator
import itertools
import numpy as np
from operator import itemgetter


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
if(input_file == 7):
    file_train = open("Data\\NBA\\new.csv")
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

#to get the distinct class from train data
def GetUniqueClass(Train):
    xxx = list(map(list, zip(*Train)))
    cls= xxx.pop(len(xxx)-1);
    #print((cls))
    for i  in range(len(cls)):
        cls[i]=float(cls[i])
    cls = set(cls)
    cls = list(cls)
    #print(cls)
    return cls

def GetCentroidList(Train, unique_cls):
    sum = 0
    cls_centroid = []
    cls_centroid_list = []
    #create sample list to calculate sum of each feature
    for i in range(len(Train[0])-1):
        cls_centroid.append(0.0)

    counter=0 #for howmany rows has particular class for division in centroid method
    for cls in unique_cls:
        for row in Train:
            if row[len(Train[0]) - 1] == cls:
                for feature in range(len(row)-1):
                    sum = cls_centroid[feature]
                    sum = sum + row[feature]
                    cls_centroid[feature] = sum
                counter=counter+1
        cls_centroid = [x / counter for x in cls_centroid]
        cls_centroid_list.append(cls_centroid)
        counter=0
        cls_centroid=[]
        for i in range(len(Train[0])-1):
            cls_centroid.append(0.0)

    #append class at the end of the centroid list
    index=0
    for row in cls_centroid_list:
        row.append(unique_cls[index])
        index=index+1

    return cls_centroid_list

def euclidean_dis(x, y):
  dist = 0.0
  for i in range(len(x) - 1):
    dist += pow((x[i] - y[i]), 2)
  dist = math.sqrt(dist)
  return dist

def methodcall(Train, Test):

    unique_cls = GetUniqueClass(Train)
    cls_centroid_list = GetCentroidList(Train, unique_cls)
    Euclidian_Matrix_row = []
    Derived_class_index = []

    for each_test_data in Test:
        # calculate euclidian distance
        for each_centroid_class in cls_centroid_list:
            Euclidian_Matrix_row.append(euclidean_dis(each_test_data, each_centroid_class))
        # extract minimum distance index from the each class that is the nearest centroid from current test data
        Derived_class_index.append(min(enumerate(Euclidian_Matrix_row), key=itemgetter(1))[0])
        Euclidian_Matrix_row = []

    counter = 0
    derived_class = []
    # get the actual class of that particular index
    for i in Derived_class_index:
        derived_class.append(cls_centroid_list[i][len(cls_centroid_list[0]) - 1])


    counter = 0
    actual_class = []
    for row in range(len(Test)):
        actual_class.append(Test[row][len(Test[0]) - 1])
        if (Test[row][len(Test[0]) - 1] == derived_class[row]):
            counter += 1

    return (counter * 100 / len(Test))



Train = []
Test = []
Answer = []
for i in range(len(temp)):
    for row in range(len(New_Train_Data)):
        if row  in temp[i]:
            Test.append(New_Train_Data[row])
        else:
            Train.append(New_Train_Data[row])
    Answer.append(methodcall(Train, Test))
    print(np.shape(Train))
    print(np.shape(Test))
    Test=[]
    Train=[]

print(Answer)
print(np.mean(Answer))
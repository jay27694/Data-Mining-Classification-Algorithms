import csv, math
import operator
import itertools
import numpy as np

#take datasets
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

New_Train_Data = ConvertData(Data_train)
New_Test_Data = ConvertData(Data_test)



def euclidean_dis(x, y):
  dist = 0.0
  for i in range(len(x) - 1):
    dist += pow((x[i] - y[i]), 2)
  dist = math.sqrt(dist)
  return dist

#get most common class from the list
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

#calculate euclidean distance matrix
Euclidean_Matrix_Row = []
Euclidean_Matrix = []
for i in range(len(New_Test_Data)):
    for j in range(len(New_Train_Data)):
        Euclidean_Matrix_Row.append(euclidean_dis(New_Test_Data[i],New_Train_Data[j]))
    Euclidean_Matrix.append(Euclidean_Matrix_Row)
    Euclidean_Matrix_Row=[]

find_repeat_list = []
derived_class = []

for row in range(len(Euclidean_Matrix)):
    #find the index of the euclidien distance in sorted order
    temp = sorted(range(len(Euclidean_Matrix[row])), key=lambda i: Euclidean_Matrix[row][i])
    # find kth nearest distance class
    for k in range(kNN_k):
        find_repeat_list.append(New_Train_Data[temp[k]][len(New_Train_Data[0])-1])
    #find common class from them
    com_cls = most_common(find_repeat_list)
    derived_class.append(com_cls)
    find_repeat_list = []

print("=============================================================================================================")
print("Derived class with kNN")
print(derived_class)
print("=============================================================================================================")
print("Actual class")
actual_class=[]
counter = 0
for row in range(len(New_Test_Data)):
    actual_class.append(New_Test_Data[row][len(New_Test_Data[0]) - 1])
    if (New_Test_Data[row][len(New_Test_Data[0])-1] == derived_class[row]):
        counter += 1
print(actual_class)
print("=============================================================================================================")
print("Accuracy in %")
print(counter*100/len(New_Test_Data))
print("=============================================================================================================")
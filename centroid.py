import csv, math
from operator import itemgetter

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

#convert data in to Features...-class format
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

#to get the distinct class from train data
def GetUniqueClass():
    cls=Data_train.pop(0);
    for i  in range(len(cls)):
        cls[i]=float(cls[i])
    cls = set(cls)
    cls = list(cls)
    return cls

#get distinct class from above method
unique_cls = GetUniqueClass()

#calculate the centroid of each unique class
def GetCentroidList(New_Train_Data):
    sum = 0
    cls_centroid = []
    cls_centroid_list = []
    #create sample list to calculate sum of each feature
    for i in range(len(New_Train_Data[0])-1):
        cls_centroid.append(0.0)

    counter=0 #for howmany rows has particular class for division in centroid method
    for cls in unique_cls:
        for row in New_Train_Data:
            if row[len(New_Train_Data[0]) - 1] == cls:
                for feature in range(len(row)-1):
                    sum = cls_centroid[feature]
                    sum = sum + row[feature]
                    cls_centroid[feature] = sum
                counter=counter+1
        cls_centroid = [x / counter for x in cls_centroid]
        cls_centroid_list.append(cls_centroid)
        counter=0
        cls_centroid=[]
        for i in range(len(New_Train_Data[0])-1):
            cls_centroid.append(0.0)

    #append class at the end of the centroid list
    index=0
    for row in cls_centroid_list:
        row.append(unique_cls[index])
        index=index+1

    return cls_centroid_list

cls_centroid_list = GetCentroidList(New_Train_Data)
#print(cls_centroid_list)
def euclidean_dis(x, y):
  dist = 0.0
  for i in range(len(x) - 1):
    dist += pow((x[i] - y[i]), 2)
  dist = math.sqrt(dist)
  return dist

Euclidian_Matrix_row=[]
Derived_class_index=[]

for each_test_data in New_Test_Data:
    #calculate euclidian distance
    for each_centroid_class in cls_centroid_list:
        Euclidian_Matrix_row.append(euclidean_dis(each_test_data,each_centroid_class))
    #extract minimum distance index from the each class that is the nearest centroid from current test data
    Derived_class_index.append(min(enumerate(Euclidian_Matrix_row), key=itemgetter(1))[0])
    Euclidian_Matrix_row=[]

counter=0
derived_class=[]
#get the actual class of that particular index
for i in Derived_class_index:
    derived_class.append(cls_centroid_list[i][len(cls_centroid_list[0])-1])

print("==========================================================================================================")
counter = 0
print("Derived class with Centroid")
print(derived_class)
print("==========================================================================================================")
print("Actual class")
actual_class=[]
for row in range(len(New_Test_Data)):
    actual_class.append(New_Test_Data[row][len(New_Test_Data[0])-1])
    if (New_Test_Data[row][len(New_Test_Data[0])-1] == derived_class[row]):
        counter += 1
print(actual_class)
print("==========================================================================================================")
print("Accuracy in %")
print(counter*100/len(New_Test_Data))
print("==========================================================================================================")
import csv
import numpy as np
import heapq

file_train = open("Data\\nba.csv")

k = int(input("How many feature you want: "))

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

#convert data format
#New_Train_Data = ConvertData(Data_train)

New_Train_Data = np.array(Data_train)


def FStat(X, Y):

    X = [float(i) for i in X]
    MeanX = np.array(X).mean()
    unique_cls = np.unique(Y)

    class_val = []
    class_val_row = []

    for i in range(len(unique_cls)):
        for j in range(len(Y)):
            if (Y[j] == unique_cls[i]):
                class_val_row.append(X[j])
        class_val.append(class_val_row)
        class_val_row = []

    def Mean_Var(Data):
        Data = np.array(Data)
        mean = Data.mean()
        a = 0.0
        for val in Data:
            a += (abs(val - mean) ** 2)
        var = a / (len(Data) - 1)
        return mean, var

    Mean = []
    Var = []

    for i in range(len(unique_cls)):
        avg, vr = Mean_Var(class_val[i])
        Mean.append(avg)
        Var.append(vr)

    F1 = 0.0
    for i in range(len(unique_cls)):
        F1 += (len(class_val[i]) * ((Mean[i] - MeanX) ** 2))
    F1 = F1 / (len(unique_cls) - 1)

    F2 = 0.0
    for i in range(len(unique_cls)):
        F2 += ((len(class_val[i]) - 1) * (Var[i]))

    F2 = F2 / (len(X) - len(unique_cls))

    F = F1 / F2
    return F

fval =[]

for col in range(len(New_Train_Data[0])-1):
    X = New_Train_Data[:, col]
    Y = New_Train_Data[:, len(New_Train_Data[0])-1]
    fval.append(FStat(X, Y))

fval = np.array(fval)
print(fval)

TakeColLargest = heapq.nlargest(k, range(len(fval)), fval.take)
TakeColSmallest = heapq.nsmallest(k, range(len(fval)), fval.take)
print(TakeColLargest)
print(TakeColSmallest)

TempData = New_Train_Data[:, TakeColLargest]

cls = New_Train_Data[:, len(New_Train_Data[0])-1]

TempData = TempData.tolist()
cls = cls.tolist()

Data =[]
c = 0

for row in TempData:
    row.append(cls[c])
    Data.append(row)
    c+=1
#print(Data)
filewrite = open('Data\\Output\\new.csv', 'w')
for row in Data:
    for i in range(len(row)):
        filewrite.write(str(row[i]))
        if (i != len(row)-1):
            filewrite.write(',')
    filewrite.write("\n")


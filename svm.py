from sklearn import svm
import csv, numpy as np

print("1: ATNT50, 2: ATNT200")
input_file = int(input("Enter dataset number: "))
if(input_file == 1):
    file_train = open("Data\\ATNT50\\csv\\trainDataXY.csv")
    file_test = open("Data\\ATNT50\\csv\\testDataXY.csv")
if(input_file == 2):
    file_train = open("Data\\ATNT200\\csv\\trainDataXY.csv")
    file_test = open("Data\\ATNT200\\csv\\testDataXY.csv")
if(input_file == 3):
    file_train = open("Data\\exam\\MTrainDataXY.csv")
    file_test = open("Data\\exam\\MTestDataX.csv")

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

#extract classes from train
Train_class = []
no = len(New_Train_Data[0])-1
for i in range(len(New_Train_Data)):
        Train_class.append(New_Train_Data[i].pop(no))

#extract classes from test
Test_class = []
no = len(New_Test_Data[0])-1
for i in range(len(New_Test_Data)):
        Test_class.append(New_Test_Data[i].pop(no))

#normlization of DATA for SVM library
def normalize(Data):
    temp = []
    Data_max = []
    Data_min = []

    #find Max and Min
    for j in range(len(Data[0])):
        for i in range(len(Data)):
            temp.append(Data[i][j])
        Data_max.append(max(temp))
        Data_min.append(min(temp))
        temp = []

    #formula = (actual value- min of that row)/(max-min)
    temp1 = []
    New_Data = []
    for j in range(len(Data[0])):
        for i in range(len(Data)):
            temp1.append((Data[i][j] - Data_min[j])/(Data_max[j]-Data_min[j]))
        New_Data.append(temp1)
        temp1 = []

    Data = map(list, zip(*New_Data))
    Data = list(Data)
    return Data

#New_Train_Data = normalize(New_Train_Data)
#New_Test_Data= normalize(New_Test_Data)

#run svm classifier
clf = svm.SVC(kernel="linear")
clf.fit(New_Train_Data, Train_class)
clf.decision_function_shape="ovo"
#print(clf.n_support_)
#print(clf.support_)

#print(len(clf.coef_))
#print(len(clf.coef_[0]))
#LinearSVC, weight vector and bias
#coef = np.array(clf.coef_)
#Test_Data = np.array(New_Test_Data)
#Test_Data = np.transpose(Test_Data)
#mul = np.matmul(coef, Test_Data)
#print(mul)
#x = mul.argmax(axis=0)
#print(x)

#alpha value in SVC
#alphas = np.abs(clf.dual_coef_)
#print(alphas)
Derived_class = clf.predict(New_Test_Data)

print("=============================================================================================================")
print("Derived class with SVM")
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
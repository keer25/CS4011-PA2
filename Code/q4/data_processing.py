from PIL import Image
import glob
import numpy as np
import csv

def features(data):
    rows = len(data)
    cols = len(data[0])
    all_features = []
    for j in range(cols):
        bins = [0]*32
        for i in range(rows):
            bins[int(data[i][j]/8)] += 1
        all_features.extend(bins)
    return all_features

def write_d(file_name, data):
    f = open(file_name, "w")
    f.write(header)
    f.write("%d %d\n"%(len(data),len(data[0])))
    for j  in range(len(data[0])):
        for i in range(len(data)):
            f.write("%d\n"%data[i][j])

# Write the data in a CSV file
# There are 96 features of the data
# def write_csv(split, features, labels):
#     data1 = np.concatenate((np.array(features), np.array(labels)), axis=1)
#     with open("../../Dataset/DS2-"+ split +".csv", "w") as f:
#         writer = csv.writer(f)
#         writer.writerows(data1)

# data_processing code
splits = ["Test", "Train"]
classes = ["mountain", "forest", "coast", "insidecity"]
labels = {}
labels["mountain"] = -1
labels["forest"] = 1
labels["coast"] = 0
labels["insidecity"] = 2
header = "%%MatrixMarket matrix array real general\n"
# for split in splits:
all_features = []
class_labels = []
for clas in classes[0:4]:
    for filename in glob.glob(clas + "/" + splits[0] + '/*.jpg'): #assuming gif
        im=Image.open(filename)
        data = im.getdata()
        all_features.append(features(data))
        class_labels.append([labels[clas]])
    for filename in glob.glob(clas + "/" + splits[1] + '/*.jpg'): #assuming gif
        im=Image.open(filename)
        data = im.getdata()
        all_features.append(features(data))
        class_labels.append([labels[clas]])
print(len(all_features))
print(len(class_labels))
data = np.concatenate((np.array(all_features), np.array(class_labels)), axis=1)
with open("../../Dataset/DS2" + ".csv", "w") as f:
    writer = csv.writer(f)
    writer.writerows(data)


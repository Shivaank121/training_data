import csv
import os
import json

csv_file_name = "train.csv"
directory_name = ['Nainital']

lines = [[]]
with open('train.csv', 'w') as writeFile:
    writer = csv.writer(writeFile)
    lines = [['Place','Review','Adventure','Pilgrimage','Spiritual and Heritage','Chill & Relax','Work & Travel','T&L','Social Tourism (Volunteer & Travel)']]
    writer.writerows(lines)

def write_in_file(lines):
    with open('train.csv', 'w') as writeFile:
        writer = csv.writer(writeFile)
        writer.writerows(lines)

def get_lable_for_file(file_name):
    print("1: Adventure\n")
    print("2: Pilgrimage\n")
    print("3: Spiritual and Heritage\n")
    print("4: Chill & Relax\n")
    print("5: Work & Travel\n")
    print("6: T&L\n")
    print("7: Social Tourism (Volunteer & Travel)\n")
    print("Enter space seprated lables for " + file_name + " :")
    lables = list(map(int, input().split()))
    return lables

def add_into_training_data(dir,file_name):
    lables = get_lable_for_file(file_name)
    reviews_file = open(dir + "/"+file_name)
    reviews_json = reviews_file.read()
    reviews_dictionary = json.loads(reviews_json)
    for index in range(0, len(reviews_dictionary)):
        print(str(index)+" \n")
        row = ['place', 'review', 0, 0, 0, 0, 0, 0, 0]
        for i in lables:
            row[i + 1] = 1
        row[0] = reviews_dictionary[index]['point_of_interest']
        row[1] = reviews_dictionary[index]['review']
        lines.append(row)
    write_in_file(lines)


for directory in directory_name :
    file_list = os.listdir(directory)
    for review_file in file_list:
        if review_file.endswith('json') :
            add_into_training_data(directory,review_file)
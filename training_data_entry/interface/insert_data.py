import os
from interface import models

city = 'Rishikesh'
directory = os.fsencode("/home/shivaank/Desktop/training_data/training_data_entry/interface/Rishikesh")
for file in os.listdir(directory):
    filename = os.fsdecode(file)
    if filename.endswith(".json"):
        # print(filename)
        name = filename[:-5]
        entry = models.POIClass.objects.create(city = city, poi = name)
        if entry:
            entry.save()
            print(name)


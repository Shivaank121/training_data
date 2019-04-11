import csv
import json
from interface.models import POIClass

ids = {
    'Adventure & Outdoors': 0,
    'Spiritual': 1,
    'Nature & Retreat': 2,
    'Isolated or Hippie': 3,
    'Heritage': 4,
    'Travel & Learn': 5,
    'Social Tourism (Volunteer & Travel)': 6,
    'Nightlife & Events': 7,
    'Shopping': 8
}

csvFile = open('data.csv', 'w')
writer = csv.writer(csvFile)
row = ['City','Place', 'Review', 'Adventure & Outdoors','Spiritual', 'Nature & Retreat', 'Isolated or Hippie',
       'Heritage', 'Travel & Learn', 'Social Tourism (Volunteer & Travel)', 'Nightlife & Events', 'Shopping',]
writer.writerow(row)

objects = POIClass.objects.filter(filters__isnull=False)
for obj in objects.distinct():
    #print(obj.poi + ": ")
    city = obj.city
    file = open( 'interface/' + city + '/' + obj.poi + '.json', 'r')
    json_data = json.load(file)
    #print (json_data)
    for r in json_data:
        row = [city, obj.poi, r["review"], 0, 0, 0, 0, 0, 0, 0, 0, 0]
        for f in obj.filters.all():
            row[ids[f.name]+3] = 1
        writer.writerow(row)

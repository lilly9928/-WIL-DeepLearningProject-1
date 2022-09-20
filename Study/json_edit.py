import json

path = "C:/Users/1315/Desktop/data/sample.json"

data = {}
data['playlist'] = []
data['playlist'].append({
    "singer": "TAEYEON",
    "song": "Weekend",
    "date": 20210706
})
data['playlist'].append({
    "singer": "almost monday",
    "song": "til the end of time",
    "date": 20210709
})

print(data)

# save json
with open(path, 'w') as outfile:
    json.dump(data, outfile, indent=4)

# load json data
with open(path, "r") as json_file:
    json_data = json.load(json_file)

print(json_data)


# insert element
json_data['playlist'].append({
    "singer": "The Volunteers",
    "song": "Summer",
    "date": 20210527
})

print(json_data)
import json

def load(path):
    with open(path, 'r') as f:
        return json.load(f)
train = 'dataset/tacred/train.json'
dev = 'dataset/tacred/dev.json'


td = load(train)
dd = load(dev)
print(len(td))

print(len(dd))
td = td + dd
print(len(td))
with open('dataset/tacred/full_train.json', 'w') as f:
    json.dump(td, f)
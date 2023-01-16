import json

dictlist = []
def read_json(file):
    data = json.load(file)
    for key, value in data.items():
       	val = list(value.values())
       	tumour = val[0]
       	total = val[1]
       	result = tumour / total
       	print('tumour cells : ', tumour, '  |  total cells : ', total, '  |  Density : ', result)

if __name__ == '__main__':
    with open("test.json") as f:
        read_json(f)
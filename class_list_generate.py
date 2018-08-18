import csv



def find_class_list(path):
    classList = []
    with open(path) as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if row['class'] not in classList:
                classList.append(row['class'])
        return classList



number = list(input())
count = [0,0]

for i in range(len(number)):
    if i == 0:
        current = number[i]
        count[int(current)] += 1

    if number[i] != current:
        current = number[i]
        count[int(current)]+= 1


print(min(count))





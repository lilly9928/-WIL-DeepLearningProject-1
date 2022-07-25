n = int(input())
numbers=[]

for i in range(n):
    numbers.append(int(input()))

def count(number):
    if number == 1:
        return 1
    elif number == 2:
        return 2
    elif number == 3:
        return 4
    else:
        return count(number - 1) + count(number - 2) + count(number - 3)

for num in numbers:
    print(count(num))

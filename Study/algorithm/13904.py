import sys

number = int(input())
arr = []

for i in range(number):
    d,w = input().split()
    d= int(d)
    w = int(w)
    arr.append([d,w])

max_date = max(t[0] for t in arr)
answer=[0 for _ in range(max_date)]

arr.sort(reverse=True,key=lambda x:x[1])

#print(arr)

for i in range(number):
    for j in range(arr[i][0]-1,-1,-1): #역순
        if answer[j] == 0:
            answer[j] = arr[i][1]
            break

print(answer)
print(sum(answer))
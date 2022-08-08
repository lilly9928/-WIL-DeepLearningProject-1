import heapq

number = int(input())

Arr =[]
myclass =[0]

for i in range(number):
    id,start,end = input().split()
    id = int(id)
    start= int(start)
    end = int(end)
    Arr.append([start,end])

heapq.heapify(Arr)

while Arr:
    start, end = heapq.heappop(Arr)
    if myclass[0] <= start:
        heapq.heappop(myclass)
    heapq.heappush(myclass, end)

print(len(myclass))


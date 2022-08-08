import heapq

n, k = map(int, input().split())
force =list(map(int, input().split()))

myforce = 0
force.sort()
heapq.heapify(force)

if n <= k:
    k = n-1

for i in range(n):
    if i == 0:
        continue
    elif i < k:
        myforce += force[i]

    else:
        myforce += force[i]*k


print(myforce)


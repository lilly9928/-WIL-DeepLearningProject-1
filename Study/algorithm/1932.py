n = int(input())

arr = [_ for _ in range(n)]

for i in range(n):
    arr[i] = list(map(int,input().split(' ')))

for i in range(1,n):
    for j in range(len(arr[i])):
        if j == 0:
            arr[i][j] += arr[i-1][j]
        elif j == i:
            arr[i][j] += arr[i-1][j-1]
        else:
            arr[i][j] += max(arr[i-1][j],arr[i-1][j-1])


print(max(max(arr)))

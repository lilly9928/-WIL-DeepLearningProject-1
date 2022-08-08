n = int(input())

arr=(list(map(int,input().split())))

tmp = []

for i in range(0,n):
    real_num = i+1

    if i ==0:
        tmp.append(arr[i]*n)
    elif real_num == n:
        tmp.append(arr[i])
    elif real_num == n-1:
        tmp.append(arr[i]+arr[0])
    else:
        multi = int(n//real_num)
        if n%real_num == 0:
            tmp.append(arr[i]*multi)
            tmp.append(arr[i]+arr[0]*(n-real_num))
        else:
            for left_num in range(0,i):
                tmp.append(arr[i]*(int(n//real_num)) + arr[left_num] * (n - real_num))


print(tmp)
print(max(tmp))



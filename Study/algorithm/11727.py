n = int(input())

count = [0,1,3]

for i in range(3,n+1):
    count.append(count[i-1]+count[i-2]*2)

answer = count[n]%10007

print(answer)

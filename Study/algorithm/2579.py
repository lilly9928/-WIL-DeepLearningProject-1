n = int(input())

stair=[]

for i in range(n):
    stair.append(int(input()))

answer = stair[n-1]
count = 1
me = n-1

for idx in range(n-2,0,-1):
    print('idx',idx,'me',me)
    if idx == me:
        continue
    if stair[idx]>stair[idx-1]:
        me = me-1
        if count ==2:
            count =0
            continue
        else:
            answer = answer+stair[idx]
            count = count + 1
            print(me,stair[idx])

    elif stair[idx]<stair[idx-1]:
        count = 0
        me = me - 2
        answer = answer + stair[idx-1]
        print(me,stair[idx])

    elif stair[idx]==stair[idx-1]:
        me = me - 1
        answer = answer+stair[idx-1]
        print(me,stair[idx])


if count != 2 and me == 1:
    answer = answer+stair[0]
    print( count,stair[0])

print(answer)
import sys
s = str(sys.stdin.readline())
prev = ''
zero = 0
one = 0

for i in range(0, (len(s)-1)):
    if (s[i] == '0') and (prev == '0'):
        prev = '0'
        continue
    elif (s[i] == '1') and (prev == '1'):
        prev = '1'
        continue
    elif (s[i] == '0') and (prev == '1'):
        zero += 1
        prev = '0'
    elif (s[i] == '1') and (prev == '0'):
        one += 1
        prev = '1'
    else:
        prev = str(s[i])
if zero == 0 and one == 0:
    print(0)
else:
    print(zero if zero<one else one)
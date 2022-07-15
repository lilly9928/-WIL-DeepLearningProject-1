x = input()
count_arr = x.split('.')

if x.count('X') %2 != 0:
    print(-1)

else:
    x = x.replace("XXXX","AAAA")
    x = x.replace("XX","BB")

    print(x)
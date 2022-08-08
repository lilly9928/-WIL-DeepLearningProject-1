sensor_n = int(input())
tower = int(input())
sensor = list(map(int,input().split()))

if tower>sensor_n:
    tower=sensor_n

length_count = []
sensor.sort()

for i in range(0,sensor_n-1):
    length_count.append(sensor[i+1]-sensor[i])

length_count.sort()

for i in range(1,tower):
    length_count.pop()

print(sum(length_count))


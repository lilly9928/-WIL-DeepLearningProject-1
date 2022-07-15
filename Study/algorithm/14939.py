graph = [list(input()) for _ in range(10)]
answer =0


def power(arr,i,j):
	if arr[i][j] == 'O':
		arr[i][j] ='#'


for i in range(0,9):
	for j in range(0,9):
		if graph[i - 1][j] == 'O' and graph[i][j+1] == 'O':
			print('n1',i, j, graph[i][j])
			if graph[i][j-1]== 'O' and graph[i][j+1] == 'O':
				answer += 1  # all down

print(graph)


print(answer)


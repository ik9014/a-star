import cv2
import numpy as np
import copy
import random
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# 이미지 파일 경로 리스트
imgList11 = [r'C:/8-puzzle/blank.png', 
             r'C:/8-puzzle/1.png', 
             r'C:/8-puzzle/2.png']

imgList12 = [r'C:/8-puzzle/3.png', 
             r'C:/8-puzzle/4.png', 
             r'C:/8-puzzle/5.png']

imgList13 = [r'C:/8-puzzle/6.png', 
             r'C:/8-puzzle/7.png', 
             r'C:/8-puzzle/8.png']

# 모든 이미지를 하나의 리스트로 결합
imgList1 = [imgList11, imgList12, imgList13]

cur = [['0', '1', '2'], ['3', '4', '5'], ['6', '7', '8']] # 초기 퍼즐 상태
goal = [['0', '1', '2'], ['3', '4', '5'], ['6', '7', '8']] # 목표 퍼즐 상태
oper = ['up', 'down', 'right', 'left'] # 가능한 이동 연산 목록

# 기능: 주어진 이동 연산에 따라 퍼즐 상태 변경
# 입력: 퍼즐, blank의 좌표, 이동방향 문자열 (up, down, left, right)
# 출력: 업데이트된 퍼즐 or None
def movePuzzle(puzzle, x, y, oper):
    
    if(oper == 'up'):
        if(x - 1 < 0):  # 이동이 불가한 상태면 None 반환
            return None
        else: # blank와 그 위의 값을 바꾼다
            tmp = puzzle[x][y] 
            puzzle[x][y] = puzzle[x-1][y]
            puzzle[x-1][y] = tmp

            return puzzle # 업데이트된 퍼즐 반환

    # 아래부터는 oper만 다를 뿐 원리는 똑같다
    elif(oper == 'down'):
        if (x + 1 >= 3):
            return None
        else:
            tmp = puzzle[x][y]
            puzzle[x][y] = puzzle[x + 1][y]
            puzzle[x + 1][y] = tmp

            return puzzle

    elif(oper == 'right'):
        if (y + 1 >= 3):
            return None
        else:
            tmp = puzzle[x][y]
            puzzle[x][y] = puzzle[x][y + 1]
            puzzle[x][y + 1] = tmp

            return puzzle

    elif(oper == 'left'):
        if (y - 1 < 0):
            return None
        else:
            tmp = puzzle[x][y]
            puzzle[x][y] = puzzle[x][y - 1]
            puzzle[x][y - 1] = tmp

            return puzzle

# 기능: 퍼즐에서 Blank의 위치를 찾는다
# 입력: 퍼즐
# 출력: Blank의 현재 좌표
def checkZero(puzzle):
    for i in range(3):
        for j in range(3):
            if puzzle[i][j] == '0':
                return i, j
    return None

# 기능: 퍼즐을 랜덤으로 섞는다
# 입력: 퍼즐, 퍼즐을 섞는 횟수
# 출력: 무작위로 섞인 퍼즐
def shufflePuzzle(puzzle, moves):
    for _ in range(moves):
        x, y = checkZero(puzzle)
        move = random.choice(oper)
        
        # Blank를 moves 횟수만큼 랜덤하게 움직여서 퍼즐을 섞는다
        movePuzzle(puzzle, x, y, move)
    return puzzle

# 퍼즐의 현재 상태와 휴리스틱 값, 레벨과 부모 노드를 저장한다
class Node:
    # 기능: 객체를 초기화하는 생성자이다
    def __init__(self, data, hval, level, parent=None):
        self.data = data 
        self.hval = hval 
        self.level = level 
        self.parent = parent 

# 기능: 현재 퍼즐과 목표 퍼즐간의 다른 타일 수 계산 (휴리스틱)
# 입력: 현재 퍼즐, 목표 퍼즐
# 출력: 휴리스틱 값
def h(puzzle, goal):
    cnt = 0
    for i in range(3):
        for j in range(3):
            if puzzle[i][j] != goal[i][j]:
                cnt += 1
    return cnt

# 기능: 특정 노드의 총 비용을 계산한다
# 입력: 현재 노드, 목표 퍼즐
# 출력: 총 비용
def f(node, goal):
    return node.level + h(node.data, goal)

# 기능: 현재 퍼즐 상태를 시각화
# 입력: 퍼즐, 3x3 그리드
# 출력: X
def visualize_puzzle(puzzle, axes):
    for i in range(3):
        for j in range(3):
            axes[i, j].cla()  # 현재 이미지를 지운다
            img = cv2.imread(imgList1[int(puzzle[i][j]) // 3][int(puzzle[i][j]) % 3])  # 이미지 로드
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # 원본 이미지 색감 유지
            axes[i, j].imshow(img)
            axes[i, j].axis('off')  # 축을 숨긴다
    plt.draw()
    plt.pause(0.1)  # 이미지 로드 시 걸리는 시간

# 기능: 퍼즐 클릭을 감지한다
# 입력: 마우스 클릭 이벤트
# 출력: X
def mouseClick(event):
    if event.inaxes:
        i, j = int(event.inaxes.rowNum), int(event.inaxes.colNum)
        x, y = checkZero(cur)
        if (abs(i - x) == 1 and j == y) or (abs(j - y) == 1 and i == x):
            cur[x][y], cur[i][j] = cur[i][j], cur[x][y]
            visualize_puzzle(cur, axes)

# 기능: 수동으로 퍼즐을 푼다
# 입력: 퍼즐
# 출력: X
def manual_puzzle(puzzle):
    global fig, axes
    fig, axes = plt.subplots(3, 3, figsize=(5, 5)) #서브플롯 생성
    plt.subplots_adjust(wspace=0.01, hspace=0.01) # 플롯 사이 간격 설정
    for i in range(3):
        for j in range(3):
            img = imgList1[int(puzzle[i][j]) // 3][int(puzzle[i][j]) % 3] # 이미지 파일경로 로드
            img2 = cv2.imread(img) # 이미지 로드
            ax = axes[i, j] # 현재 플롯 선택
            ax.imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)) # 원본 이미지 색감 유지
            ax.axis('off') # 축을 숨긴다
            ax.rowNum = i
            ax.colNum = j
    # 마우스 클릭 이벤트와 연결
    fig.canvas.mpl_connect('button_press_event', mouseClick) 
    plt.show()

# 기능: A* 알고리즘 이용하여 퍼즐 해결
# 입력: 초기 퍼즐 상태
# 출력: 방문한 노드의 리스트
def automatic_puzzle(puzzle):
    visit = set() # 방문한 노드 저장
    queue = [] # 탐색할 노드 저장
    start = Node(data=puzzle, hval=h(puzzle, goal), level=0) # 초기 퍼즐 상태
    queue.append(start)

    fig, axes = plt.subplots(3, 3, figsize=(5, 5)) # 3x3 서브플롯 생성
    plt.subplots_adjust(wspace=0.01, hspace=0.01) # 서브플롯 사이 간격 설정

    while queue: # 큐가 빌 때까지 실행
        current = queue.pop(0) # 큐의 앞에 있는 노드를 꺼낸다
        print(np.array(current.data))

        # 목표 상태에 도달하면 탐색 종료
        if h(current.data, goal) == 0:
            path = []
            while current:
                path.append(current)
                current = current.parent
            path.reverse()
            for node in path:
                visualize_puzzle(node.data, axes)
                if node.level == 0:
                    plt.pause(1)
                plt.pause(0.1)
            plt.show()
            return visit

        visit.add(tuple(map(tuple, current.data))) # 현재 노드를 방문배열에 저장
        x, y = checkZero(current.data)

        for op in oper:
            next_puzzle = copy.deepcopy(current.data) # 퍼즐을 복사한다
            next = movePuzzle(next_puzzle, x, y, op) # 빈 타일을 이동시킨다
            if next and tuple(map(tuple, next)) not in visit: # 새로운 퍼즐이 유효하고 아직 방문하지 않은 경우
                queue.append(Node(next, h(next, goal), current.level + 1, current))
                visit.add(tuple(map(tuple, next)))
        queue.sort(key=lambda x: f(x, goal)) # 평가값 기준 큐 정렬, 작은 값 순으로 노드 탐색

    return -1

cur = shufflePuzzle(cur, 1000)
automatic_puzzle(cur)

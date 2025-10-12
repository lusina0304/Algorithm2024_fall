# -*- coding: utf-8 -*-
"""
Queue와 Deque 구현 및 BFS 미로 찾기 실습
자료구조 수업용 예제 코드
"""
from typing import List, Tuple, Optional


class cir_queue:
    def __init__(self, size):
        ## 전역 변수 선언 부분 ##
        self.size = size
        self.queue = [None for _ in range(size)]
        self.front = 0
        self.rear = 0

    ## 함수 선언 부분 ##
    def isQueueFull(self):
        if ((self.rear + 1) % self.size == self.front):
            return True
        else:
            return False

    def isQueueEmpty(self):
        if (self.front == self.rear):
            return True
        else:
            return False

    def enQueue(self, data):
        if (self.isQueueFull()):
            print("큐가 꽉 찼습니다.")
            return False
        self.rear = (self.rear + 1) % self.size
        self.queue[self.rear] = data
        return True

    def deQueue(self):
        if (self.isQueueEmpty()):
            print("큐가 비었습니다.")
            return None
        self.front = (self.front + 1) % self.size
        data = self.queue[self.front]
        self.queue[self.front] = None
        return data

    def peek(self):
        if (self.isQueueEmpty()):
            print("큐가 비었습니다.")
            return None
        return self.queue[(self.front + 1) % self.size]



# ==================== BFS 미로 찾기 ====================
class MazeSolver:
    """
    2D 그리드 미로를 BFS로 해결하는 클래스

    미로 표현:
    - 0: 벽 (지나갈 수 없음)
    - 1: 길 (지나갈 수 있음)
    - S: 시작점
    - E: 탈출점
    """

    def __init__(self, maze: List[List[int]], start: Tuple[int, int], end: Tuple[int, int]):
        """
        Args:
            maze: 2D 그리드 미로 (0: 벽, 1: 길)
            start: 시작점 좌표 (row, col)
            end: 탈출점 좌표 (row, col)
        """
        self.maze = maze
        self.rows = len(maze)
        self.cols = len(maze[0]) if maze else 0
        self.start = start
        self.end = end

        # 상하좌우 이동 방향
        self.directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        self.direction_names = ['UP', 'DOWN', 'LEFT', 'RIGHT']

    def is_valid(self, row: int, col: int, visited: set) -> bool:
        """
        현재 위치가 유효한지 확인
        - 그리드 범위 내에 있는지
        - 벽이 아닌지
        - 방문하지 않았는지
        """
        return (0 <= row < self.rows and
                0 <= col < self.cols and
                self.maze[row][col] == 1 and
                (row, col) not in visited)

    def bfs_with_custom_queue(self, queue_size=1000) -> Optional[List[Tuple[int, int]]]:
        """
        직접 구현한 cir_queue를 사용한 BFS

        Args:
            queue_size: 원형 큐의 최대 크기 (기본값: 1000)

        Returns:
            경로를 나타내는 좌표 리스트, 경로가 없으면 None
        """
        queue = cir_queue(queue_size)
        queue.enQueue((self.start, [self.start]))  # (현재 위치, 경로)
        visited = {self.start}

        while not queue.isQueueEmpty():
            item = queue.deQueue()
            if item is None:  # 큐가 비었을 때
                break

            (row, col), path = item

            # 탈출점 도착
            if (row, col) == self.end:
                return path

            # 4방향 탐색
            for dr, dc in self.directions:
                new_row, new_col = row + dr, col + dc

                if self.is_valid(new_row, new_col, visited):
                    visited.add((new_row, new_col))
                    new_path = path + [(new_row, new_col)]

                    # 큐가 꽉 찼을 경우 처리
                    if not queue.enQueue(((new_row, new_col), new_path)):
                        print(f"Warning: Queue is full at queue_size={queue_size}")
                        print("Consider increasing queue_size parameter")
                        return None

        return None  # 경로를 찾지 못함

    def print_maze(self, path: Optional[List[Tuple[int, int]]] = None):
        """
        미로를 시각적으로 출력
        경로가 주어지면 경로를 표시
        """
        path_set = set(path) if path else set()

        print("\n[Maze]")
        for i in range(self.rows):
            row_str = ""
            for j in range(self.cols):
                if (i, j) == self.start:
                    row_str += "S "
                elif (i, j) == self.end:
                    row_str += "E "
                elif (i, j) in path_set:
                    row_str += "* "
                elif self.maze[i][j] == 0:
                    row_str += "# "
                else:
                    row_str += ". "
            print(row_str)
        print()


# ==================== 데모 및 테스트 ====================
def demo_circular_queue():
    """Circular Queue 사용 예제"""
    print("=" * 50)
    print("Circular Queue Demo")
    print("=" * 50)

    # 크기 5의 원형 큐 생성
    q = cir_queue(5)
    print(f"Empty queue - Is empty? {q.isQueueEmpty()}")

    print("\nEnqueuing: 1, 2, 3")
    for i in range(1, 4):
        q.enQueue(i)
        print(f"enQueue({i}) -> queue: {q.queue}, front={q.front}, rear={q.rear}")

    print(f"\nPeek: {q.peek()}")

    print("\nDequeuing 2 items:")
    for _ in range(2):
        item = q.deQueue()
        print(f"deQueue() -> {item}, queue: {q.queue}, front={q.front}, rear={q.rear}")

    print("\nEnqueuing: 4, 5, 6")
    for i in range(4, 7):
        q.enQueue(i)
        print(f"enQueue({i}) -> queue: {q.queue}, front={q.front}, rear={q.rear}")

    print("\nDequeuing all:")
    while not q.isQueueEmpty():
        item = q.deQueue()
        print(f"deQueue() -> {item}, remaining: {q.queue}")

    print()


def demo_maze_solver():
    """BFS 미로 찾기 예제"""
    print("=" * 50)
    print("BFS Maze Solver Demo")
    print("=" * 50)

    # 미로 정의 (0: 벽, 1: 길)
    maze = [
        [1, 1, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 0, 1, 1, 1],
        [0, 0, 0, 1, 0, 1, 0, 0],
        [1, 1, 0, 1, 1, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0, 1],
        [0, 1, 1, 1, 0, 1, 1, 1],
        [0, 0, 0, 1, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 1, 1]
    ]

    start = (0, 0)  # 시작점
    end = (7, 7)    # 탈출점

    solver = MazeSolver(maze, start, end)

    # 원본 미로 출력
    print("\nLegend: S=Start, E=End, #=Wall, .=Path, *=Solution")
    solver.print_maze()

    # 직접 구현한 cir_queue를 사용한 BFS
    print("\nBFS using custom Circular Queue:")
    path = solver.bfs_with_custom_queue(queue_size=100)
    if path:
        print(f"Path found! (length: {len(path)})")
        print(f"Path coordinates: {path}")
        solver.print_maze(path)
    else:
        print("No path found.")


def demo_difficult_maze():
    """더 복잡한 미로 예제"""
    print("\n" + "=" * 50)
    print("Complex Maze Example")
    print("=" * 50)

    # 더 복잡한 미로
    maze = [
        [1, 0, 1, 1, 1, 1, 0, 1, 1, 1],
        [1, 0, 1, 0, 0, 1, 0, 0, 0, 1],
        [1, 1, 1, 0, 0, 1, 1, 1, 0, 1],
        [0, 0, 0, 0, 1, 0, 0, 1, 0, 1],
        [1, 1, 1, 1, 1, 0, 0, 1, 0, 1],
        [1, 0, 0, 0, 1, 1, 1, 1, 0, 1],
        [1, 0, 1, 0, 0, 0, 0, 0, 0, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    ]

    start = (0, 0)
    end = (7, 9)

    solver = MazeSolver(maze, start, end)

    print("\nLegend: S=Start, E=End, #=Wall, .=Path, *=Solution")
    solver.print_maze()

    # 더 큰 미로이므로 큐 크기를 크게 설정
    path = solver.bfs_with_custom_queue(queue_size=200)
    if path:
        print(f"Path found! (length: {len(path)})")
        solver.print_maze(path)
    else:
        print("No path found.")


def demo_fibonacci():
    """피보나치 수열을 큐를 이용해 계산하는 예제"""
    print("\n" + "=" * 50)
    print("Fibonacci Sequence using Circular Queue")
    print("=" * 50)

    print("\n피보나치 수열: F(n) = F(n-1) + F(n-2)")
    print("F(0) = 0, F(1) = 1")

    n = 10
    print(f"\n첫 {n}개의 피보나치 수를 큐를 이용해 계산:")

    # 크기 3의 큐 생성 (F(n-2), F(n-1)만 저장하면 됨)
    q = cir_queue(3)

    # 초기값 설정
    q.enQueue(0)  # F(0)
    q.enQueue(1)  # F(1)

    print(f"F(0) = 0")
    print(f"F(1) = 1")

    for i in range(2, n):
        # 큐에서 가장 오래된 값을 제거
        f_n_2 = q.deQueue()

        # 큐의 맨 앞 값 확인 (F(n-1))
        f_n_1 = q.peek()

        # 새로운 피보나치 수 계산
        f_n = f_n_2 + f_n_1

        # 새 값을 큐에 추가
        q.enQueue(f_n)

        print(f"F({i}) = {f_n}")

    print("\n" + "=" * 50)
    print("다른 방법: 재귀와 메모이제이션")
    print("=" * 50)

    # 메모이제이션을 위한 딕셔너리
    memo = {}

    def fibonacci_recursive(n):
        """재귀를 이용한 피보나치 (메모이제이션)"""
        if n in memo:
            return memo[n]

        if n <= 1:
            return n

        memo[n] = fibonacci_recursive(n - 1) + fibonacci_recursive(n - 2)
        return memo[n]

    print("\n재귀로 계산한 피보나치 수:")
    for i in range(n):
        print(f"F({i}) = {fibonacci_recursive(i)}")

    print("\n" + "=" * 50)
    print("반복문을 이용한 방법 (가장 효율적)")
    print("=" * 50)

    def fibonacci_iterative(n):
        """반복문을 이용한 피보나치"""
        if n <= 1:
            return n

        a, b = 0, 1
        for _ in range(2, n + 1):
            a, b = b, a + b
        return b

    print("\n반복문으로 계산한 피보나치 수:")
    for i in range(n):
        print(f"F({i}) = {fibonacci_iterative(i)}")

    print("\n" + "=" * 50)
    print("큐를 이용한 피보나치 수열 생성기")
    print("=" * 50)

    def fibonacci_generator_with_queue(count):
        """큐를 이용한 피보나치 제너레이터"""
        q = cir_queue(3)
        q.enQueue(0)
        q.enQueue(1)

        yield 0
        yield 1

        for _ in range(2, count):
            f_n_2 = q.deQueue()
            f_n_1 = q.peek()
            f_n = f_n_2 + f_n_1
            q.enQueue(f_n)
            yield f_n

    print(f"\n제너레이터로 생성한 첫 {n}개의 피보나치 수:")
    fib_sequence = list(fibonacci_generator_with_queue(n))
    print(fib_sequence)

    print("\n" + "=" * 50)
    print("성능 비교 (n=30)")
    print("=" * 50)

    import time

    # 큐 방식
    start = time.time()
    q = cir_queue(3)
    q.enQueue(0)
    q.enQueue(1)
    for i in range(2, 30):
        f_n_2 = q.deQueue()
        f_n_1 = q.peek()
        f_n = f_n_2 + f_n_1
        q.enQueue(f_n)
    end = time.time()
    print(f"큐 방식: {(end - start) * 1000:.6f} ms")

    # 재귀 방식 (메모이제이션)
    memo.clear()
    start = time.time()
    result = fibonacci_recursive(30)
    end = time.time()
    print(f"재귀 방식 (메모이제이션): {(end - start) * 1000:.6f} ms")

    # 반복문 방식
    start = time.time()
    result = fibonacci_iterative(30)
    end = time.time()
    print(f"반복문 방식: {(end - start) * 1000:.6f} ms")

    print(f"\nF(30) = {result}")


def main():
    """메인 함수"""
    print("\n" + "=" * 50)
    print("Data Structures: Circular Queue & BFS")
    print("=" * 50 + "\n")

    # Circular Queue 데모
    demo_circular_queue()

    # BFS 미로 찾기 데모
    demo_maze_solver()

    # 복잡한 미로 데모
    demo_difficult_maze()

    # 피보나치 수열 데모
    demo_fibonacci()

    print("\n" + "=" * 50)
    print("Demo Complete!")
    print("=" * 50)


if __name__ == "__main__":
    main()

# 그림 코드
def print_grid(grid):
    for row in grid:
        line = ""
        for cell in row:
            if cell == 0:
                line += "■ "
            elif cell == 1:
                line += "□ "
            elif cell == 2:
                line += "x "
            elif cell == 5:
                line += "★ "
            else:
                line += "? "
        print(line)

def fill(y, x, color, targetColor):
    if grid[y][x] != color or grid[y][x] == targetColor:
        return

    grid[y][x] = targetColor

    for ny, nx in [(y + 1, x), (y - 1, x), (y, x + 1), (y, x - 1)]:
        if not (0 <= ny < len(grid)) or not (0 <= nx < len(grid[0])):
            continue

        fill(ny, nx, color, targetColor) 


grid = [
    [0,1,1,1,1,1],
    [2,0,1,1,0,0],
    [2,0,1,0,0,1],
    [2,0,0,1,1,1],
    [0,1,0,1,1,1],
    [1,1,0,1,1,1],
]

print("=== Before fill ===")
print_grid(grid)

fill(1, 0, grid[1][0], 5)

print("\n=== After fill ===")
print_grid(grid)
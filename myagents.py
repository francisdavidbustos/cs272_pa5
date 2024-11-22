from random import randint
import ourhexenv

class MyDumbAgent():
    def __init__(self, env) -> None:
        self.env = env

    def place(self) -> int:
        xVal = randint(0, self.env.board_size - 1)
        yVal = randint(0, self.env.board_size - 1)

        while self.env.board[xVal][yVal] != 0:
            xVal = randint(0, self.env.board_size - 1)
            yVal = randint(0, self.env.board_size - 1)

        return xVal * self.env.board_size + yVal

    def swap(self) -> int:
        return randint(0,1)

class MyABitSmarterAgent():
    def __init__(self, env) -> None:
        self.env = env
        self.visited = set()
        self.start = None

    def place(self) -> int:
        if self.start is None:
            return self.begin()
        temp = self.dfs()
        self.start = (temp // self.env.board_size, temp % self.env.board_size)
        return temp

    def swap(self) -> int:
        return randint(0,1)

    def begin(self) -> int:

        xVal = randint(0, self.env.board_size - 1)
        yVal = randint(0, self.env.board_size - 1)

        while self.env.board[xVal][yVal] != 0:
            xVal = randint(0, self.env.board_size - 1)
            yVal = randint(0, self.env.board_size - 1)

        self.start = (xVal, yVal)
        self.visited.add(self.start)

        return xVal * self.env.board_size + yVal

    def dfs(self) -> int:

        steps = [(-1,0), (1,0), (0,-1), (0,1), (-1,-1), (-1,1), (1,-1), (1,1)]
        temp = [self.start]
        while temp:
            x,y = temp.pop()
        
            for stepX, stepY in steps:
                newX, newY = x + stepX, y + stepY
                if 0 <= newX < self.env.board_size and 0 <= newY < self.env.board_size and self.env.board[newX][newY] == 0:
                    if (newX, newY) not in self.visited:
                        temp.append((newX, newY))
                        self.visited.add((newX,newY))
                        return newX * self.env.board_size + newY

        return self.adj()

    def adj(self) -> int:
        x, y = self.start
        steps = [(-1,0), (1,0), (0,-1), (0,1), (-1,-1), (-1,1), (1,-1), (1,1)]

        for stepX, stepY in steps:
            newX, newY = x + stepX, y + stepY
            if 0 <= newX < self.env.board_size and 0 <= newY < self.env.board_size and self.env.board[newX][newY] == 0:
                if (newX, newY) not in self.visited:
                    self.visited.add((newX, newY))
                    return newX * self.env.board_size + newY

        return self.begin()
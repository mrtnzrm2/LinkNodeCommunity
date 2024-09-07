import random

class Node:
  def __init__(self, key, column) -> None:
    self.key = key
    self.left = None
    self.right = None
    self.column = column
  
  def display(self):
      lines, *_ = self._display_aux()
      for line in lines:
          print(line)

  def _display_aux(self):
      """Returns list of strings, width, height, and horizontal coordinate of the root."""
      # No child.
      if self.right is None and self.left is None:
          line = '%s' % self.key
          width = len(line)
          height = 1
          middle = width // 2
          return [line], width, height, middle

      # Only left child.
      if self.right is None:
          lines, n, p, x = self.left._display_aux()
          s = '%s' % self.key
          u = len(s)
          first_line = (x + 1) * ' ' + (n - x - 1) * '_' + s
          second_line = x * ' ' + '/' + (n - x - 1 + u) * ' '
          shifted_lines = [line + u * ' ' for line in lines]
          return [first_line, second_line] + shifted_lines, n + u, p + 2, n + u // 2

      # Only right child.
      if self.left is None:
          lines, n, p, x = self.right._display_aux()
          s = '%s' % self.key
          u = len(s)
          first_line = s + x * '_' + (n - x) * ' '
          second_line = (u + x) * ' ' + '\\' + (n - x - 1) * ' '
          shifted_lines = [u * ' ' + line for line in lines]
          return [first_line, second_line] + shifted_lines, n + u, p + 2, u // 2

      # Two children.
      left, n, p, x = self.left._display_aux()
      right, m, q, y = self.right._display_aux()
      s = '%s' % self.key
      u = len(s)
      first_line = (x + 1) * ' ' + (n - x - 1) * '_' + s + y * '_' + (m - y) * ' '
      second_line = x * ' ' + '/' + (n - x - 1 + u + y) * ' ' + '\\' + (m - y - 1) * ' '
      if p < q:
          left += [n * ' '] * (q - p)
      elif q < p:
          right += [m * ' '] * (p - q)
      zipped_lines = zip(left, right)
      lines = [first_line, second_line] + [a + u * ' ' + b for a, b in zipped_lines]
      return lines, n + m + u, max(p, q) + 2, n + u // 2


class Tree:
    def __init__(self) -> None:
        self.root = None
        self.min_column = 0
        self.max_column = 0
        self.n = 0

    def insert(self, node, key, column):
        # If the tree is empty, return a new node
        if column > self.max_column: self.max_column = column
        if column < self.min_column: self.min_column = column

        if node is None:
            return Node(key, column)

        r = random.randint(0, 2)
        # Otherwise, recur down the tree
        if r == 1:
            node.left = self.insert(node.left, key, column - 1)
        else:
            node.right = self.insert(node.right, key, column + 1)

        self.n += 1
        return node
 
    def dfs(self, root : Node, column):
        # Base Cases: root is null or key is present at root
        if root.key == column:
            return root
        if root.left is not None:
          self.dfs(root.left, column)
        if root.right is not None:
          self.dfs(root.right, column)

if  __name__ == "__main__":
   
    T = Tree()
    T.root = T.insert(T.root, 2, 0)
    T.root = T.insert(T.root, 4, 0)
    T.root = T.insert(T.root, 5, 0)
    T.root = T.insert(T.root, 6, 0)
    T.root = T.insert(T.root, 7, 0)
    T.root = T.insert(T.root, 8, 0)
    T.root = T.insert(T.root, 11, 0)
    T.root = T.insert(T.root, 20, 0)
    T.root = T.insert(T.root, 9, 0)

    if isinstance(T.root, Node):
      T.root.display()

    column_nodes = {}
    def printColumnNodes(root, column):
        if root is None:
            return
        if root.column == column:
            if column not in column_nodes.keys():
                column_nodes[column] = [root.key]
            else:
                column_nodes[column].append(root.key)

        if isinstance(root.left, Node): printColumnNodes(root.left, column)
        if isinstance(root.right, Node): printColumnNodes(root.right, column)

    for c in range(T.min_column, T.max_column+1):
        printColumnNodes(T.root, c)

    print(column_nodes)
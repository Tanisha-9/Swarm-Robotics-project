import random
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# ------------------ GRID GRAPH ------------------

class GridGraph:
    def __init__(self, rows, cols):
        self.rows = rows
        self.cols = cols
        self.nodes = [(i, j) for i in range(rows + 1)
                             for j in range(cols + 1)]

# ------------------ ROBOT ------------------

class Robot:
    def __init__(self, robot_id, position):
        self.robot_id = robot_id
        self.position = position
        self.rank = None

    def __repr__(self):
        return f"Robot {self.robot_id} at {self.position} (rank={self.rank})"

# ------------------ SYMMETRY CHECK ------------------

def reflect_horizontal(positions, rows):
    return {(rows - x, y) for (x, y) in positions}

def reflect_vertical(positions, cols):
    return {(x, cols - y) for (x, y) in positions}

def rotate_180(positions, rows, cols):
    return {(rows - x, cols - y) for (x, y) in positions}

def is_symmetric(positions, rows, cols):
    pos_set = set(positions)
    return (
        pos_set == reflect_horizontal(pos_set, rows)
        or pos_set == reflect_vertical(pos_set, cols)
        or pos_set == rotate_180(pos_set, rows, cols)
    )

# ------------------ ASYMMETRIC PLACEMENT ------------------

def place_robots_asymmetric(grid, num_robots):
    anchor = (0, 0)

    for _ in range(500):
        remaining = list(set(grid.nodes) - {anchor})
        sampled = random.sample(remaining, num_robots - 1)
        positions = [anchor] + sampled

        if not is_symmetric(positions, grid.rows, grid.cols):
            return [Robot(i + 1, pos) for i, pos in enumerate(positions)]

    raise RuntimeError("Failed to find asymmetric placement")

# ------------------ BLOCK NODES PLACEMENT ------------------

def get_4_neighbors(position, grid):
    """Get 4-connected neighbors (up, down, left, right)"""
    x, y = position
    neighbors = []
    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        nx, ny = x + dx, y + dy
        if 0 <= nx <= grid.rows and 0 <= ny <= grid.cols:
            neighbors.append((nx, ny))
    return neighbors

def is_connected(grid, blocked_nodes, robot_positions):
    """
    Check if grid remains connected after blocking nodes using BFS.
    Returns True if all non-blocked nodes are reachable from a starting point.
    """
    # Walkable nodes = all nodes - blocked nodes
    walkable = set(grid.nodes) - set(blocked_nodes)
    
    if not walkable:
        return False
    
    # Start BFS from first robot position (guaranteed to be walkable)
    start = robot_positions[0]
    visited = {start}
    queue = [start]
    
    # BFS to explore all reachable nodes
    while queue:
        current = queue.pop(0)
        
        # Check all 4 neighbors
        for neighbor in get_4_neighbors(current, grid):
            if neighbor in walkable and neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
    
    # Grid is connected if all walkable nodes are visited
    return len(visited) == len(walkable)

def place_blocks(grid, num_blocks, robot_positions):
    """
    Randomly place block nodes ensuring grid remains connected.
    """
    if num_blocks == 0:
        return set()
    
    # Available nodes = all nodes - robot positions
    available = set(grid.nodes) - set(robot_positions)
    
    if num_blocks > len(available):
        raise ValueError(f"Cannot block {num_blocks} nodes. Only {len(available)} available.")
    
    # Try multiple times to find valid blocking
    for attempt in range(500):
        # Randomly select num_blocks nodes
        blocked = set(random.sample(list(available), num_blocks))
        
        # Check if grid remains connected
        if is_connected(grid, blocked, robot_positions):
            return blocked
    
    raise RuntimeError("Failed to find valid block placement that keeps grid connected")

# ------------------ RANK COMPUTATION ------------------

def traversal_orders(grid):
    R, C = grid.rows, grid.cols
    return [
        [(x, y) for x in range(0, R + 1) for y in range(0, C + 1)],
        [(x, y) for x in range(0, R + 1) for y in range(C, -1, -1)],
        [(x, y) for x in range(R, -1, -1) for y in range(0, C + 1)],
        [(x, y) for x in range(R, -1, -1) for y in range(C, -1, -1)]
    ]

def compute_ranks(grid, robots):
    positions = {r.position for r in robots}
    robot_map = {r.position: r for r in robots}

    best_string = ""
    best_order = None

    for order in traversal_orders(grid):
        s = ''.join('1' if n in positions else '0' for n in order)
        rs = s[::-1]

        if s > best_string:
            best_string = s
            best_order = order
        if rs > best_string:
            best_string = rs
            best_order = list(reversed(order))

    rank = 1
    for node in best_order:
        if node in robot_map:
            robot_map[node].rank = rank
            rank += 1

# ------------------ VISUALIZATION (MODIFIED TO SHOW BLOCKED NODES) ------------------

def plot_grid(grid, robots, blocked_nodes, show_ranks=False, filename="output.png", title=""):
    fig, ax = plt.subplots(figsize=(6, 6))

    # Grid lines
    for x in range(grid.rows + 1):
        ax.plot([0, grid.cols], [x, x], linewidth=1)
    for y in range(grid.cols + 1):
        ax.plot([y, y], [0, grid.rows], linewidth=1)

    robot_positions = {r.position: r for r in robots}

    # Squares at ALL nodes
    for (x, y) in grid.nodes:
        px = y
        py = grid.rows - x

        size = 0.28
        
        # Determine square color
        if (x, y) in blocked_nodes:
            # Gray square for blocked nodes
            facecolor = 'gray'
        else:
            # White square for normal nodes
            facecolor = 'white'
        
        square = Rectangle(
            (px + 0.12, py - 0.12),
            size,
            size,
            linewidth=0.8,
            edgecolor='black',
            facecolor=facecolor,
            zorder=2
        )
        ax.add_patch(square)

        # Show ranks only on non-blocked nodes with robots
        if show_ranks and (x, y) in robot_positions and (x, y) not in blocked_nodes:
            ax.text(
                px + 0.12 + size / 2,
                py - 0.12 + size / 2,
                str(robot_positions[(x, y)].rank),
                ha='center',
                va='center',
                fontsize=9,
                fontweight='bold',
                zorder=4
            )

    # Small black circles on blocked nodes
    for (x, y) in blocked_nodes:
        ax.scatter(
            y,
            grid.rows - x,
            s=80,
            c='black',
            zorder=3
        )
    
    # Robots
    for r in robots:
        ax.scatter(
            r.position[1],
            grid.rows - r.position[0],
            s=350,
            edgecolors='black',
            linewidths=1.5,
            zorder=5
        )

    ax.set_xlim(-0.3, grid.cols + 0.6)
    ax.set_ylim(-0.3, grid.rows + 0.6)
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title)

    plt.savefig(filename, dpi=200, bbox_inches="tight")
    plt.close()

# ------------------ SAVE STATE ------------------

def save_state(grid, robots, blocked_nodes, filename="state.json"):
    data = {
        "rows": grid.rows,
        "cols": grid.cols,
        "blocked_nodes": [list(pos) for pos in blocked_nodes],
        "robots": [
            {
                "id": r.robot_id,
                "position": list(r.position),
                "rank": r.rank
            } for r in robots
        ]
    }
    with open(filename, "w") as f:
        json.dump(data, f, indent=4)

# ------------------ MAIN ------------------

if __name__ == "__main__":

    rows = int(input("Enter number of rows (cells): "))
    cols = int(input("Enter number of columns (cells): "))
    num_robots = int(input("Enter number of robots: "))
    num_blocks = int(input("Enter number of nodes to block: "))

    grid = GridGraph(rows, cols)
    
    # Place robots first
    robots = place_robots_asymmetric(grid, num_robots)
    robot_positions = [r.position for r in robots]
    
    # Place block nodes (avoiding robot positions, maintaining connectivity)
    blocked_nodes = place_blocks(grid, num_blocks, robot_positions)
    
    print(f"Placed {len(blocked_nodes)} blocked nodes: {blocked_nodes}")

    plot_grid(grid, robots, blocked_nodes, False, "before_ranking.png", "Initial Configuration")

    compute_ranks(grid, robots)

    plot_grid(grid, robots, blocked_nodes, True, "after_ranking.png", "After Symmetry Breaking")

    save_state(grid, robots, blocked_nodes)

    print("Generated:")
    print(" - before_ranking.png")
    print(" - after_ranking.png")
    print(" - state.json")
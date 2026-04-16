import random
import json
import heapq
from collections import deque, defaultdict
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
    def __init__(self, robot_id, position, rank):
        self.robot_id = robot_id
        self.position = position
        self.rank = rank

    def __repr__(self):
        return f"Robot {self.robot_id} at {self.position} (rank={self.rank})"

# ------------------ VISUALIZATION (UNCHANGED) ------------------

def plot_grid(grid, robots, marked_positions, blocked_nodes, filename="output.png", title=""):
  
    fig, ax = plt.subplots(figsize=(6, 6))

    # Grid lines
    for x in range(grid.rows + 1):
        ax.plot([0, grid.cols], [x, x], 'k-', linewidth=1)
    for y in range(grid.cols + 1):
        ax.plot([y, y], [0, grid.rows], 'k-', linewidth=1)

    # Draw all squares (marked with ranks OR blocked in gray)
    for (x, y) in grid.nodes:
        px = y
        py = grid.rows - x
        size = 0.28

        # Determine square appearance
        if (x, y) in blocked_nodes:
            # Gray square for blocked nodes
            square = Rectangle(
                (px + 0.12, py - 0.12),
                size,
                size,
                linewidth=0.8,
                edgecolor='black',
                facecolor='gray',
                zorder=2
            )
            ax.add_patch(square)
        elif (x, y) in marked_positions:
            # White square with rank for marked positions
            square = Rectangle(
                (px + 0.12, py - 0.12),
                size,
                size,
                linewidth=0.8,
                edgecolor='black',
                facecolor='white',
                zorder=2
            )
            ax.add_patch(square)

            ax.text(
                px + 0.12 + size / 2,
                py - 0.12 + size / 2,
                str(marked_positions[(x, y)]),
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

    # Draw current robot positions
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

# ------------------ MOVEMENT LOGIC ------------------

def get_neighbors(position, grid, blocked_nodes):
   
    x, y = position
    neighbors = []
    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        nx, ny = x + dx, y + dy
        if (0 <= nx <= grid.rows and 0 <= ny <= grid.cols
                and (nx, ny) not in blocked_nodes):
            neighbors.append((nx, ny))
    return neighbors

def is_occupied(position, robots):
    return any(r.position == position for r in robots)

def is_marked(position, marked_positions):
    return position in marked_positions

def get_robot_at(position, robots):
    for r in robots:
        if r.position == position:
            return r
    return None

def categorize_neighbor(w, u, robots, marked_positions, blocked_nodes, grid):
    w_neighbors = get_neighbors(w, grid, blocked_nodes)
    w_neighbors = [n for n in w_neighbors if n != u]

    has_occupied_neighbor = False
    has_free_marked_neighbor = False

    for wn in w_neighbors:
        if is_occupied(wn, robots) or wn in blocked_nodes:
            has_occupied_neighbor = True
        if not is_occupied(wn, robots) and wn not in blocked_nodes and is_marked(wn, marked_positions):
            has_free_marked_neighbor = True

    if not has_occupied_neighbor and not has_free_marked_neighbor:
        return 1
    elif has_occupied_neighbor:
        return 2
    elif has_free_marked_neighbor:
        return 3

    return None

def compute_forward_position(robot, robots, marked_positions, blocked_nodes, grid):
 
    u = robot.position
    neighbors = get_neighbors(u, grid, blocked_nodes)

    case1_neighbors = []
    case2_neighbors = []
    case3_neighbors = []

    for w in neighbors:
        if not is_marked(w, marked_positions):
            case = categorize_neighbor(w, u, robots, marked_positions, blocked_nodes, grid)
            if case == 1:
                case1_neighbors.append(w)
            elif case == 2:
                case2_neighbors.append(w)
            elif case == 3:
                case3_neighbors.append(w)

    # Try Case 1 first
    if case1_neighbors:
        marked_case1 = [(w, marked_positions.get(w, -1)) for w in case1_neighbors]
        max_rank = max(rank for _, rank in marked_case1)
        if max_rank == -1:
            return random.choice(case1_neighbors)
        else:
            candidates = [w for w, rank in marked_case1 if rank == max_rank]
            return random.choice(candidates) if len(candidates) > 1 else candidates[0]

    # Try Case 2
    elif case2_neighbors:
        for w1 in case2_neighbors:
            w1_neighbors = get_neighbors(w1, grid, blocked_nodes)
            w1_neighbors = [n for n in w1_neighbors if n != u]
            neighbor_robot_ranks = []
            for wn in w1_neighbors:
                neighbor_robot = get_robot_at(wn, robots)
                if neighbor_robot:
                    neighbor_robot_ranks.append(neighbor_robot.rank)
            if not neighbor_robot_ranks or robot.rank > max(neighbor_robot_ranks):
                return w1

    # Try Case 3
    else:
        for w1 in case3_neighbors:
            w1_neighbors = get_neighbors(w1, grid, blocked_nodes)
            w1_neighbors = [n for n in w1_neighbors if n != u]
            free_marked_ranks = []
            for wn in w1_neighbors:
                if (not is_occupied(wn, robots) and wn not in blocked_nodes
                        and is_marked(wn, marked_positions)):
                    free_marked_ranks.append(marked_positions[wn])
            if not free_marked_ranks or robot.rank >= max(free_marked_ranks):
                return w1

    # No valid move
    return u

# ------------------ BACKTRACK LOGIC ------------------

def bfs_nearest_unmarked(start, grid, blocked_nodes, occupied_positions, marked_positions, excluded_targets=None):
   
    if excluded_targets is None:
        excluded_targets = set()

    visited = {start}
    # Queue entries: (current_position, distance, first_step)
    # first_step = the immediate next node from start (None until we take first step)
    queue = deque()
    queue.append((start, 0, None))

    while queue:
        current, dist, first_step = queue.popleft()

        # Check if current is a valid unmarked target (not the start itself)
        if (current != start
                and current not in marked_positions
                and current not in excluded_targets):
            return current, dist, first_step

        # Expand neighbors
        for neighbor in get_neighbors(current, grid, blocked_nodes):
            if neighbor not in visited:
                # Cannot pass through other robots (they block the path)
                if neighbor in occupied_positions:
                    visited.add(neighbor)
                    continue
                visited.add(neighbor)
                # Track first step from start
                step = first_step if first_step is not None else neighbor
                queue.append((neighbor, dist + 1, step))

    return None, float('inf'), None


def resolve_backtrack_targets(robots_backtracking, grid, blocked_nodes, all_robots, marked_positions):
  
    robot_bfs = {}
    for robot in robots_backtracking:
        occupied = {r.position for r in all_robots if r.robot_id != robot.robot_id}
        candidates = []
        visited = {robot.position}
        queue = deque()
        queue.append((robot.position, 0, None))

        while queue:
            current, dist, first_step = queue.popleft()
            if current != robot.position and current not in marked_positions:
                candidates.append((dist, current, first_step))
            for neighbor in get_neighbors(current, grid, blocked_nodes):
                if neighbor not in visited:
                    if neighbor in occupied:
                        visited.add(neighbor)
                        continue
                    visited.add(neighbor)
                    step = first_step if first_step is not None else neighbor
                    queue.append((neighbor, dist + 1, step))

        candidates.sort(key=lambda x: x[0])
        robot_bfs[robot.robot_id] = candidates

    claimed_targets = {}
    next_steps = {}
    assigned_distances = {}

    robot_assigned = {r.robot_id: False for r in robots_backtracking}
    robot_pointer = {r.robot_id: 0 for r in robots_backtracking}
    rank_map = {r.robot_id: r.rank for r in robots_backtracking}

    heap = []
    for robot in robots_backtracking:
        candidates = robot_bfs[robot.robot_id]
        if candidates:
            dist, target, step = candidates[0]
            heapq.heappush(heap, (dist, -rank_map[robot.robot_id], robot.robot_id, target, step))

    while heap and not all(robot_assigned.values()):
        dist, neg_rank, rid, target, step = heapq.heappop(heap)

        if robot_assigned[rid]:
            continue

        if target in claimed_targets:
            robot_pointer[rid] += 1
            candidates = robot_bfs[rid]
            while robot_pointer[rid] < len(candidates):
                d, t, s = candidates[robot_pointer[rid]]
                if t not in claimed_targets:
                    heapq.heappush(heap, (d, -rank_map[rid], rid, t, s))
                    break
                robot_pointer[rid] += 1
        else:
            claimed_targets[target] = rid
            next_steps[rid] = step
            assigned_distances[rid] = dist
            robot_assigned[rid] = True

    for robot in robots_backtracking:
        if robot.robot_id not in next_steps:
            next_steps[robot.robot_id] = robot.position
            assigned_distances[robot.robot_id] = 999999

    return next_steps, assigned_distances


# ------------------ MAIN CYCLE ------------------

def run_simulation(grid, robots, marked_positions, blocked_nodes):
   
    # Total explorable nodes = all nodes - blocked nodes
    explorable_nodes = set(grid.nodes) - blocked_nodes
    cycle = 0
    generated_files = []

    # Save initial state image
    filename = f"cycle_000_initial.png"
    plot_grid(grid, robots, marked_positions, blocked_nodes, filename, "Initial State (Cycle 0)")
    generated_files.append(filename)
    print(f"Cycle 0 (initial): {len(marked_positions)} nodes marked")

    while True:
        # Check termination: all explorable nodes marked
        if explorable_nodes.issubset(marked_positions.keys()):
            print(f"\nGrid exploration completed successfully in {cycle} cycles")
            break

        cycle += 1

        # --- Step 1: Mark current positions of all robots ---
        for robot in robots:
            if robot.position not in marked_positions:
                marked_positions[robot.position] = robot.rank

        # --- Step 2: Determine next position for each robot ---
        # First try forward phase for all robots
        forward_results = {}
        for robot in robots:
            next_pos = compute_forward_position(robot, robots, marked_positions, blocked_nodes, grid)
            forward_results[robot.robot_id] = next_pos

        # Identify robots that are stuck (forward phase returns same position)
        stuck_robots = [r for r in robots if forward_results[r.robot_id] == r.position]
        moving_robots = [r for r in robots if forward_results[r.robot_id] != r.position]

        # --- Step 3: Resolve backtrack for stuck robots ---
        backtrack_next_steps = {}
        backtrack_distances = {}
        if stuck_robots:
            backtrack_next_steps, backtrack_distances = resolve_backtrack_targets(
                stuck_robots, grid, blocked_nodes, robots, marked_positions
            )

        # --- Step 4: Build final next positions ---
        intended = {}

        for robot in moving_robots:
            intended[robot.robot_id] = forward_results[robot.robot_id]

        for robot in stuck_robots:
            intended[robot.robot_id] = backtrack_next_steps.get(robot.robot_id, robot.position)

        # Collision resolution:
        #   - Forward beats backtrack always
        #   - Forward vs Forward: higher rank wins
        #   - Backtrack vs Backtrack: closer (smaller distance) wins; ties → higher rank
        forward_robot_ids = {r.robot_id for r in moving_robots}
        stuck_robot_ids = {r.robot_id for r in stuck_robots}
        rank_map = {r.robot_id: r.rank for r in robots}

        target_map = defaultdict(list)
        for robot in robots:
            target_pos = intended[robot.robot_id]
            if target_pos != robot.position:
                target_map[target_pos].append(robot.robot_id)

        blocked_moves = set()
        for target_pos, contender_ids in target_map.items():
            if len(contender_ids) > 1:
                fwd = [rid for rid in contender_ids if rid in forward_robot_ids]
                bck = [rid for rid in contender_ids if rid in stuck_robot_ids]

                if fwd:
                    # All backtrack robots lose
                    for rid in bck:
                        blocked_moves.add(rid)
                    # Among forward robots: highest rank wins
                    if len(fwd) > 1:
                        fwd.sort(key=lambda rid: rank_map[rid], reverse=True)
                        for rid in fwd[1:]:
                            blocked_moves.add(rid)
                else:
                    # All backtrack: closer wins; ties → higher rank wins
                    bck.sort(key=lambda rid: (backtrack_distances.get(rid, 999999), -rank_map[rid]))
                    for rid in bck[1:]:
                        blocked_moves.add(rid)

        # --- Step 5: Apply moves ---
        for robot in robots:
            if robot.robot_id not in blocked_moves:
                robot.position = intended[robot.robot_id]
            # else: robot stays in current position

        # --- Step 6: Save image for this cycle ---
        filename = f"cycle_{cycle:03d}.png"
        plot_grid(grid, robots, marked_positions, blocked_nodes, filename, f"Cycle {cycle}")
        generated_files.append(filename)

        # Print status
        mode_info = []
        for robot in robots:
            mode = "FWD" if robot.robot_id not in [r.robot_id for r in stuck_robots] else "BCK"
            mode_info.append(f"R{robot.robot_id}({mode})")
        print(f"Cycle {cycle}: {len(marked_positions)} nodes marked | " + " ".join(mode_info))

        # Safety cap to prevent infinite loops during testing
        if cycle >= 10000:
            print("Warning: Reached cycle limit of 10000. Stopping.")
            break

    print("\nGenerated images:")
    for f in generated_files:
        print(f"  - {f}")

    return generated_files


# ------------------ LOAD STATE ------------------

def load_state(filename="state.json"):
    with open(filename, "r") as f:
        data = json.load(f)

    grid = GridGraph(data["rows"], data["cols"])
    robots = [
        Robot(r["id"], tuple(r["position"]), r["rank"])
        for r in data["robots"]
    ]
    blocked_nodes = {tuple(pos) for pos in data.get("blocked_nodes", [])}
    marked_positions = {tuple(r["position"]): r["rank"] for r in data["robots"]}

    return grid, robots, marked_positions, blocked_nodes


# ------------------ MAIN ------------------

if __name__ == "__main__":
    grid, robots, marked_positions, blocked_nodes = load_state("state.json")

    print(f"Loaded {len(robots)} robots from state.json")
    print(f"Grid: {grid.rows}x{grid.cols}")
    print(f"Blocked nodes: {len(blocked_nodes)}")
    print(f"Total explorable nodes: {len(set(grid.nodes) - blocked_nodes)}")
    print()

    run_simulation(grid, robots, marked_positions, blocked_nodes)
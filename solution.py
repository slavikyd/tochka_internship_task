import sys
import heapq


COSTS = {'A': 1, 'B': 10, 'C': 100, 'D': 1000}
TARGET_ROOM_POS = {'A': 2, 'B': 4, 'C': 6, 'D': 8}
ROOM_INDICES = [2, 4, 6, 8]
INVALID_HALL_POS = set([2, 4, 6, 8])
HALL_LEN = 11
ROOM_DEPTH = 2


def data_input(lines: list[str]):
    """
    Parses the labyrinth input lines to create the initial labyrinth state.

    Args:
        lines (list[str]): List of strings representing the labyrinth input.

    Returns:
        tuple:
            corridor (tuple): tuple representing spaces in the hallway.
            rooms (tuple): tuple of tuples - the room objects top and bottom.
    """
    corridor = tuple('.' for _ in range(HALL_LEN))

    top = [lines[2][3], lines[2][5], lines[2][7], lines[2][9]]
    bottom = [lines[3][3], lines[3][5], lines[3][7], lines[3][9]]

    rooms = tuple((top[i], bottom[i]) for i in range(4))
    return (corridor, rooms)


def is_completed(state):
    """
    Checks if the labyrinth is completed.

    Args:
        state (tuple): Current labyrinth state.

    Returns:
        bool: True if all rooms contain only their respective target objects, else False.
    """
    corridor, rooms = state
    for i, t in enumerate(['A', 'B', 'C', 'D']):
        if any(obj != t for obj in rooms[i]):
            return False
    return True


def room_open(obj, room):
    """
    Checks if a room is open.

    Args:
        obj (str): The object.
        room (tuple): The current objects of a room.

    Returns:
        bool: True if the room has only '.' or the specified obj, meaning it can accept that obj.
    """
    return all(o == '.' or o == obj for o in room)


def moves_to_target(state, room_id):
    """
    Generates all valid moves of an object from a specified room to position in the corridor.

    Args:
        state (tuple): Current labyrinth state.
        room_id (int): Index of the source room.

    Returns:
        list of tuples: Each tuple contains (new_state, energy_cost) representing a move.
    """
    corridor, rooms = state
    room = rooms[room_id]
    room_pos = ROOM_INDICES[room_id]

    if all(r == '.' for r in room):
        return []
    if room_open('ABCD'[room_id], room) and all(
        r == 'ABCD'[room_id] or r == '.' for r in room
    ):
        return []

    obj_idx = 0
    while obj_idx < ROOM_DEPTH and room[obj_idx] == '.':
        obj_idx += 1
    if obj_idx == ROOM_DEPTH:
        return []

    obj = room[obj_idx]

    results = []
    for pos in range(room_pos - 1, -1, -1):
        if corridor[pos] != '.':
            break
        if pos in INVALID_HALL_POS:
            continue
        steps = obj_idx + 1 + abs(pos - room_pos)
        energy = steps * COSTS[obj]
        new_corridor = list(corridor)
        new_corridor[pos] = obj
        new_rooms = [list(r) for r in rooms]
        new_rooms[room_id][obj_idx] = '.'
        results.append(
            ((tuple(new_corridor), tuple(tuple(r) for r in new_rooms)), energy)
        )

    for pos in range(room_pos + 1, HALL_LEN):
        if corridor[pos] != '.':
            break
        if pos in INVALID_HALL_POS:
            continue
        steps = obj_idx + 1 + abs(pos - room_pos)
        energy = steps * COSTS[obj]
        new_corridor = list(corridor)
        new_corridor[pos] = obj
        new_rooms = [list(r) for r in rooms]
        new_rooms[room_id][obj_idx] = '.'
        results.append(
            ((tuple(new_corridor), tuple(tuple(r) for r in new_rooms)), energy)
        )

    return results


def moves_from_hallway(state, hall_pos):
    """
    Generates all valid moves of an object from the corridor to its target room.

    Args:
        state (tuple): Current labyrinth state (corridor, rooms).
        hall_pos (int): The corridor position of the object to move.

    Returns:
        list of tuples
    """
    corridor, rooms = state
    obj = corridor[hall_pos]
    if obj == '.':
        return []

    target_room_id = 'ABCD'.index(obj)
    target_room_pos = ROOM_INDICES[target_room_id]
    target_room = rooms[target_room_id]

    if not room_open(obj, target_room):
        return []

    start = min(hall_pos, target_room_pos) + 1
    end = max(hall_pos, target_room_pos)

    if any(corridor[pos] != '.' for pos in range(start, end)):
        return []

    depth = ROOM_DEPTH - 1
    while depth >= 0 and target_room[depth] != '.':
        depth -= 1
    if depth < 0:
        return []

    steps = abs(hall_pos - target_room_pos) + depth + 1
    energy = steps * COSTS[obj]

    new_corridor = list(corridor)
    new_corridor[hall_pos] = '.'
    new_rooms = [list(r) for r in rooms]
    new_rooms[target_room_id][depth] = obj

    return [((tuple(new_corridor), tuple(tuple(r) for r in new_rooms)), energy)]


def neighbors(state):
    """
    Generates all possible next states from the current state with respective costs.

    Args:
        state (tuple): Current labyrinth state (corridor, rooms).

    Returns:
        list of tuples: Each tuple contains (next_state, energy_cost) representing valid next moves.
    """
    result = []
    corridor, rooms = state
    for i in range(4):
        for new_state, cost in moves_to_target(state, i):
            result.append((new_state, cost))
    for pos in range(HALL_LEN):
        for new_state, cost in moves_from_hallway(state, pos):
            result.append((new_state, cost))
    return result


def heuristic(state):
    """
    Heuristic function estimating minimum remaining energy cost to solve from current state.

    Args:
        state (tuple): labyrinth state (corridor, rooms).

    Returns:
        int: minimal energy cost to solve.
    """
    corridor, rooms = state
    h = 0
    for pos, obj in enumerate(corridor):
        if obj == '.':
            continue
        target = TARGET_ROOM_POS[obj]
        dist = abs(pos - target) + ROOM_DEPTH
        h += dist * COSTS[obj]

    for room_id, room in enumerate(rooms):
        target_obj = 'ABCD'[room_id]
        for depth, obj in enumerate(room):
            if obj == '.':
                continue
            if obj != target_obj:
                dist = depth + 1
                dist += abs(ROOM_INDICES[room_id] - TARGET_ROOM_POS[obj])
                dist += ROOM_DEPTH
                h += dist * COSTS[obj]
    return h


def a_asterisk_search(start, goal_test):
    """
    Performs an A* search to find the minimal energy solution from start to end.

    Args:
        start (tuple): initial state of the labyrinth.
        goal_test (func): A func that returns True when a state is the goal.

    Returns:
        int or None: min energy to reach room or None if unsolvable.
    """
    lst = []
    heapq.heappush(lst, (0 + heuristic(start), 0, start))
    cost_so_far = {start: 0}

    while lst:
        _, cost, current = heapq.heappop(lst)
        if goal_test(current):
            return cost

        for next_state, step_cost in neighbors(current):
            new_cost = cost + step_cost
            if next_state not in cost_so_far or new_cost < cost_so_far[next_state]:
                cost_so_far[next_state] = new_cost
                priority = new_cost + heuristic(next_state)
                heapq.heappush(lst, (priority, new_cost, next_state))

    return None


def solve(lines: list[str]) -> int:
    """
    Task solver function.

    Args:
        lines (list[str]): The input lines describing the labyrinth.

    Returns:
        int: The minimal energy required to solve.
    """
    start = data_input(lines)
    result = a_asterisk_search(start, is_completed)
    return result if result is not None else 0


def main():
    """
    Main func.
    """
    lines = []
    for line in sys.stdin:
        lines.append(line.rstrip('\n'))
    result = solve(lines)
    print(result)


if __name__ == '__main__':
    main()

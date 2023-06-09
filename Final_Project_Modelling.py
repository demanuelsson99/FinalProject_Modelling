import numpy as np
import matplotlib.pyplot as plt
import imageio

"""
NOTE: To get script to run, create a folder named "frames" in the same folder as the script.

------------------------------------------To summarize the code:-------------------------------------------------
I generate a rectangular matrix. The edges of the matrix are given the value "500", which makes them "walls".
The cells "inside" the matrix are (initially) given the value "0" and represent the floor.
Walls (cells with value 500) are also added in the middle of the matrix, which makes a wall that separates
the two dance floors. This wall has a door (signified by the value 1).

The working principle: People will be more likely to move towards cells with a low value.
This script will give higher values to cells close to walls and cells around people. This makes people want
to move away from the wall and away from other people (at least they won't want to stand directly next to them)

When it starts to rain, I add a gradient to the room to the right (this is the "outside" dance floor). Cells
that are far away from the door will have higher values. This makes people want to move towards the door.

People cannot go to a cell where there is already a person or a wall.

I make TWO grids.
 - One (grid_room) to keep track of the "room". A.k.a., what values each cell has.
 - The other one (grid_people) has the same dimensions as the first, but it keeps
 track of where there are people. A person is signified by the value "100". The other cells have the value "0".

 The door will have a value of 5. The cell to the right and left of the door should have a calue of 4 when it is not "raining". This makes
 it so that people hopefully don't want to stand *in* the door. This changes when it starts to rain, and the value just to the
 right of the door is around 6. But if you move the door around/change its size, you
 migth have to manually make sure that the cells to the right have the value 6. Or don't, to see what happens
 -------------------------------------Where in the code to look---------------------------------------------
 Information about the door is in the following places:
  - make_walls() the door is "created" here
  - start_rain() the door is indirectly used here. The gradient of the right room is based on the location of the door
  - floor_update() the value of the door is reinforced at the end of this function
"""


def create_frame(t, grid):  # For animation.
    fig = plt.figure(figsize=(6, 6))
    plotgrid = grid[t, :, :]
    cmap = plt.get_cmap('Greys', N)
    plt.imshow(plotgrid, cmap=cmap, vmin=0, vmax=N - 1)

    plt.title(f'CA Configuration, t = {t}')
    plt.savefig(f'frames/img_{t}.png',
                transparent=False,
                facecolor='white'
                )
    plt.close()


def make_walls(n, m, T):  # This one is currently hard-coded (the room size)
    # Set grid
    grid = np.zeros([T, n, m])

    # Walls
    grid[:, :, 0] = 500
    grid[:, :, -1] = 500
    grid[:, 0, :] = 500
    grid[:, -1, :] = 500
    # Wall to separate the rooms
    grid[:, :, int((m - 1) / 2)] = 500
    # Door
    for d in range(door_size + 1):  # adding cells around middle door cell
        grid[:, int((n - 1) / 2) - d, int((m - 1) / 2)] = 5  # door
        grid[:, int((n - 1) / 2) + d, int((m - 1) / 2)] = 5  # door

    return grid


def initialize_floor(grid, t):  # assigns basic values to floor cells
    # Iterate through cells that are not a wall or door and assign a value to them
    for column in range(1, int((m - 1) / 2)):  # Room to the left (inside)
        for row in range(1, n - 1):
            grid[t, row, column] = 0

    # To the left of the door: Assign higher values to make people move away from it.
    door_row = int((n - 1) / 2)
    door_column = int((m - 1) / 2)
    grid_room[t, door_row - 1 - door_size:door_row +
              2 + door_size, door_column - 1] = 2
    grid_room[t, door_row - 1 - door_size:door_row +
              2 + door_size, door_column - 2] = 1

    for column in range(int((m - 1) / 2 + 1), m - 1):  # Room to the right (outside)
        for row in range(1, n - 1):
            grid[t, row, column] = 1  # assign value
    return grid


# Create a new grid (with same dimensions as old grid) to keep track of where there are people,
def insert_people(k_L, k_R):
    # k_L are the initial amount of people in the left room and k_R in the right room

    # Set grid
    grid = np.zeros([T, n, m])
    floor_space = (n - 2) * (m - 2) - n + 1

    P = (k_L + k_R)  # Total amount of people, Left room + Right room
    prob = P / floor_space  # probability of tile being occupied

    P_R = 0  # Count for people in right room
    P_L = 0  # Count for people in left room
    Tot = 0  # Total count

    while P != Tot:  # Keep iterating until the rooms have their assigned number of people

        if P_L < k_L:  # If the room have its assigned amount of people, don't insert more people here
            # Iterate through cells that are not occupied by a wall. Use "prob" to decide if the cell is occupied by a person
            for column in range(1, int((m - 1) / 2)):  # Room to the left (inside)
                for row in range(1, n - 1):
                    # If the room hasn't reached its assigned amount of people and the current cell doesn't already has a person
                    if P_L < k_L and grid[0, row, column] == 0:
                        if np.random.random() <= prob:
                            # there's a person here now
                            grid[0, row, column] = 100
                            P_L += 1  # Add 1 to the count of people here

        if P_R < k_R:
            # Room to the right (outside)
            for column in range(int((m - 1) / 2 + 1), m - 1):
                for row in range(1, n - 1):
                    if P_R < k_R and grid[0, row, column] == 0:
                        if np.random.random() <= prob:
                            # there's a person here now
                            grid[0, row, column] = 100
                            P_R += 1

        Tot = P_L + P_R  # Calculate the total amount of people

    return grid


def start_rain(grid_room, t):  # make it start to rain
    # Iterate through cells that are not a wall and add a value to them. Do nothing to the inside-room.
    for column in range(1, int((m - 1) / 2)):  # Room to the left (inside)
        for row in range(1, n - 1):
            grid_room[t, row, column] = 0

    # To the left of the door: Assign higher values to make people move away from it.
    door_row = int((n - 1) / 2)
    door_column = int((m - 1) / 2)
    grid_room[t, door_row - 1 - door_size:door_row +
              2 + door_size, door_column - 1] = 2
    grid_room[t, door_row - 1 - door_size:door_row +
              2 + door_size, door_column - 2] = 1

    for column in range(int((m - 1) / 2 + 1), m - 1):  # Room to the right (outside)
        for row in range(1, n - 1):
            door_row_upper = ((n - 1) / 2) - door_size
            door_row_lower = ((n - 1) / 2) + door_size
            door_column = (m - 1) / 2

            row_dist = (abs(row - door_row_upper) +
                        abs(row - door_row_lower)) / 2
            # lower value closer to door. Make people want to go inside
            distance = 6 + sum([row_dist, abs(column - door_column)]) / 7
            # maximum value: 6 + 6.14 = 12.14
            grid_room[t, row, column] = distance  # assign value

    return grid_room


# people move where cell-value is lowest. But won't move on top of someone else
def move_people(grid_people, grid_room, t):
    # Iterate through cells and do stuff if it is occupied with a person
    for column in range(1, m - 1):
        for row in range(1, n - 1):
            # if the cell is occupied...
            if grid_people[t, row, column] == 100:
                # find a next cell to move to (a and b are indexes)
                [a, b] = nbhd_check(grid_room, grid_people, t, row, column)
                # a and b can be the same as "row" and "column" if the person is completely surrounded.
                # remove person from the "old" cell
                grid_people[t, row, column] = 0
                grid_people[t, a, b] = 50  # add person to new cell
                # value "50" so that they aren't moved again in the same iteration. Will be changed back to 100 later.
    grid_people[t, :, :] = grid_people[t, :, :] * 2  # return all values to 100
    return grid_people


# Decides where a person will move by checking its neighbourhood
def nbhd_check(grid_room, grid_people, t, row, column):
    directions = {}  # Create a dictionary where the keys will be index for a cell, and the values will be the value of that cell
    # only add possible direction if there is no-one else there and no wall
    if grid_people[t, row + 1, column] < 50 and grid_room[t, row + 1, column] < 500:
        directions[str([row + 1, column])] = grid_room[t,
                                                       row + 1, column]  # ----north
    # only add possible direction if there is no-one else there
    if grid_people[t, row + 1, column + 1] < 50 and grid_room[t, row + 1, column + 1] < 500:
        directions[str([row + 1, column + 1])] = grid_room[t,
                                                           row + 1, column + 1]  # north-east
    # only add possible direction if there is no-one else there
    if grid_people[t, row, column + 1] < 50 and grid_room[t, row, column + 1] < 500:
        directions[str([row, column + 1])] = grid_room[t,
                                                       row, column + 1]  # ----east
    # only add possible direction if there is no-one else there
    if grid_people[t, row - 1, column + 1] < 50 and grid_room[t, row - 1, column + 1] < 500:
        directions[str([row - 1, column + 1])] = grid_room[t,
                                                           row - 1, column + 1]  # south-east
    # only add possible direction if there is no-one else there
    if grid_people[t, row - 1, column] < 50 and grid_room[t, row - 1, column] < 500:
        directions[str([row - 1, column])] = grid_room[t,
                                                       row - 1, column]  # ----south
    # only add possible direction if there is no-one else there
    if grid_people[t, row - 1, column - 1] < 50 and grid_room[t, row - 1, column - 1] < 500:
        directions[str([row - 1, column - 1])] = grid_room[t,
                                                           row - 1, column - 1]  # south-west
    # only add possible direction if there is no-one else there
    if grid_people[t, row, column - 1] < 50 and grid_room[t, row, column - 1] < 500:
        directions[str([row, column - 1])] = grid_room[t,
                                                       row, column - 1]  # ----west
    # only add possible direction if there is no-one else there
    if grid_people[t, row + 1, column - 1] < 50 and grid_room[t, row + 1, column - 1] < 500:
        directions[str([row + 1, column - 1])] = grid_room[t,
                                                           row + 1, column - 1]  # north-west
    # I will sort this list to find the cell with the lowest value, But first...

    # Check that directions is not empty
    if len(directions) == 0:
        # if person is surrounded by people and/or walls, they will remain in place
        return [row, column]
    else:
        # shuffle directions, otherwise people tend to go in a
        # certain direction if many cells have the same value
        keys = list(directions.keys())
        np.random.shuffle(keys)
        direc = {}
        for key in keys:
            direc[key] = directions[key]
        directions = direc
        # This makes people move randomly if the cell values around them are all the same

        # now, to sort the directions:
        # smallest values end up first in dictionary
        directions = dict(sorted(directions.items(), key=lambda item: item[1]))
        res = list(list(directions.keys()))[0]  # find cell with smallest value
        res = eval(res)
        a = res[0]  # first index of new cell
        b = res[1]  # second index of new cell
        return [a, b]  # return index to cell of smallest value


# Cells around people and walls will increase in value
def floor_update(grid_room, grid_people, t):
    # during rain: Maximal cell value is 10.14.
    # Iterate through cells
    for column in range(0, m):
        for row in range(0, n):
            # make people stay away from walls by increasing the value of cells near walls
            if grid_room[t, row, column] >= 500:
                try:  # "Try" and "except" since I'm technically looking outside of the matrix here.
                    grid_room[t, row + 1, column] += 3
                except:
                    d = 1  # nonsense
                try:
                    grid_room[t, row - 1, column] += 3
                except:
                    d = 1
                try:
                    grid_room[t, row, column + 1] += 3
                except:
                    d = 1
                try:
                    grid_room[t, row, column - 1] += 3
                except:
                    d = 1
            # increase values of cells around a person
            if grid_people[t, row, column] == 100:
                grid_room[t, row + 1, column] += 0.2
                grid_room[t, row + 1, column + 1] += 0.2
                grid_room[t, row, column + 1] += 0.2
                grid_room[t, row - 1, column + 1] += 0.2
                grid_room[t, row - 1, column] += 0.2
                grid_room[t, row - 1, column - 1] += 0.2
                grid_room[t, row, column - 1] += 0.2
                grid_room[t, row + 1, column - 1] += 0.2
                grid_room[t, row, column] += 0.2
    # reinforce walls and door. If a wall gets the value "501" for example,
    # the rest of the script may not recognize it as a wall...
    grid_room[:, :, 0] = 500
    grid_room[:, :, -1] = 500
    grid_room[:, 0, :] = 500
    grid_room[:, -1, :] = 500
    grid_room[:, :, int((m - 1) / 2)] = 500

    for d in range(door_size + 1):
        grid_room[:, int((n - 1) / 2) - d, int((m - 1) / 2)] = 5  # door
        grid_room[:, int((n - 1) / 2) + d, int((m - 1) / 2)] = 5  # door

    return grid_room


def count_R(grid_people, t):  # counts amount of people in room to the right
    pers = 0
    for column in range(int((m - 1) / 2 + 1), m - 1):  # Room to the right (outside)
        for row in range(1, n - 1):
            # if there's a person here, add one to the count
            if grid_people[t, row, column] == 100:
                pers += 1
    return pers


# "Fuse" the two grids so that we can plot them.
def plot_sort(grid_room, grid_people, T):
    grid = grid_room.copy()
    for t in range(0, T):
        for i in range(0, n):
            for j in range(0, m):
                if grid_people[t, i, j] == 100:
                    grid[t, i, j] = 100
    return grid


#----------------Input parameters----------------------#
N = 20  # number of states
e = 2  # number of excited states
n = 30 + 1  # grid size vertically. Important that this is uneven.
m = 30 + 30 + 1  # grid size horisontally. Important that this is uneven.
T = 300  # amount of time steps
k_L = 100  # amount of people in left room
k_R = 400  # amount of people in right room
door_size = 0  # amount of extra open cells on each side of middle open door


# Initialize two grids. One grid to describe the room layout.
# One grid to keep track of where the people are
# create two rooms that are separated by a wall (with a door)
grid_room = make_walls(n, m, T)
# initialize the "floor", a.k.a. decide what values those cells should have at t=0
grid_room = initialize_floor(grid_room, 0)
# NEW grid that keeps track of where there are people
grid_people = insert_people(k_L, k_R)

# keeps track of amount of people in the right room
pers_count = [count_R(grid_people, 0)]
T_half = 0
# increase cell-values around walls and people. To make people stay away
grid_room = floor_update(grid_room, grid_people, 0)
for t in range(1, T):
    # initially, grids for this time step will be identical to the grids in the past step
    grid_room[t, :, :] = grid_room[t - 1, :, :].copy()
    grid_people[t, :, :] = grid_people[t - 1, :, :].copy()
    grid_people = move_people(grid_people, grid_room, t)  # make people move

    if t >= 10:  # it will rain during this period
        grid_room = start_rain(grid_room, t)  # re-initialize
    if t < 10:  # it will not rain during this period
        grid_room = initialize_floor(grid_room, t)  # re-initialize

    # re-apply values around walls and people (the increased
    grid_room = floor_update(grid_room, grid_people, t)
    # values are otherwise erased by start_rain()) or initialize_floor() )
    pers_count.append(count_R(grid_people, t))
    if pers_count[-1] <= pers_count[0] / 2 and T_half == 0:
        T_half = t
print(f"Half time T_half = {T_half}")


# "fuses" the two grids (grid of rooms and grid of people) so that
grid_plot = plot_sort(grid_room, grid_people, T)
# they can be plotted

# Create an animated .gif
frames = []
for t in range(0, T):
    create_frame(t, grid_plot)
    image = imageio.v2.imread(f'frames/img_{t}.png')
    frames.append(image)

imageio.mimsave('./animated_CA.gif',  # output gif
                frames,          # array of input frames
                duration=0.5)         # optional: frames per second

# plot people in right side of room as a function of time
plt.plot(range(0, T), pers_count)
plt.xlabel('t')
plt.ylabel('Amount of people')
plt.title("Amount of people in right room as function of time")
plt.show()

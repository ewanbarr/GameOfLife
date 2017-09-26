import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation
from profilehooks import profile
import weave
from weave import converters
from numba import jit

PATTERNS = {}

PATTERNS["glider_gun"] =\
[[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0],
 [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0],
 [0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,1,1],
 [0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,1,1],
 [1,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
 [1,1,0,0,0,0,0,0,0,0,1,0,0,0,1,0,1,1,0,0,0,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0],
 [0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0],
 [0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
 [0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]]

PATTERNS["diehard"] = [[0, 0, 0, 0, 0, 0, 1, 0],
           [1, 1, 0, 0, 0, 0, 0, 0],
           [0, 1, 0, 0, 0, 1, 1, 1]]

PATTERNS["boat"] = [[1, 1, 0],
        [1, 0, 1],
        [0, 1, 0]]

PATTERNS["r_pentomino"] = [[0, 1, 1],
               [1, 1, 0],
               [0, 1, 0]]

PATTERNS["beacon"] = [[0, 0, 1, 1],
          [0, 0, 1, 1],
          [1, 1, 0, 0],
          [1, 1, 0, 0]]

PATTERNS["acorn"] = [[0, 1, 0, 0, 0, 0, 0],
         [0, 0, 0, 1, 0, 0, 0],
         [1, 1, 0, 0, 1, 1, 1]]

PATTERNS["spaceship"] = [[0, 0, 1, 1, 0],
             [1, 1, 0, 1, 1],
             [1, 1, 1, 1, 0],
             [0, 1, 1, 0, 0]]

PATTERNS["block_switch_engine"] = [[0, 0, 0, 0, 0, 0, 1, 0],
                       [0, 0, 0, 0, 1, 0, 1, 1],
                       [0, 0, 0, 0, 1, 0, 1, 0],
                       [0, 0, 0, 0, 1, 0, 0, 0],
                       [0, 0, 1, 0, 0, 0, 0, 0],
                       [1, 0, 1, 0, 0, 0, 0, 0]]

PATTERNS["glider"] = [[1, 0, 0],
          [0, 1, 1],
          [1, 1, 0]]

PATTERNS["pulsar"] = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
          [0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0],
          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
          [0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0],
          [0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0],
          [0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0],
          [0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0],
          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
          [0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0],
          [0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0],
          [0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0],
          [0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0],
          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
          [0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0],
          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]


def init(size, pattern=None):
    assert len(size) == 2, "size parameter must be a 2-tuple"
    if pattern is None:
        board = np.random.randint(0,2,size).astype('int32')
    else:
        pattern = np.asarray(pattern)
        board = np.zeros(size,dtype='int32')
        assert board.shape[0] > pattern.shape[0], "pattern has more rows than board" 
        assert board.shape[1] > pattern.shape[1], "pattern has more columns than board"
        row_margin = (board.shape[0] - pattern.shape[0])/2
        col_margin = (board.shape[1] - pattern.shape[1])/2
        board[row_margin:row_margin+pattern.shape[0],col_margin:col_margin+pattern.shape[1]] = pattern
    return board.astype('int32')

def find_neighbours(board):
    neighbours = np.zeros_like(board)
    steps = [-1,0,1]
    for a in steps:
        for b in steps:
            if (a,b) != (0,0): 
                neighbours += np.roll(np.roll(board,a,axis=0),b,axis=1)
    return neighbours

@jit
def find_neighbours_jit(board):
    neighbours = np.zeros_like(board)
    steps = [-1,0,1]
    for a in steps:
        for b in steps:
            if (a,b) != (0,0):
                neighbours += np.roll(np.roll(board,a,axis=0),b,axis=1)
    return neighbours

def find_neighbours_weave(board):
    nx, ny = board.shape
    neighbours = np.zeros_like(board)
    code = """
    for (int ii=0; ii < ny; ++ii)
    {
        for (int jj=0; jj < nx; ++jj)
        {
            if ((ii==0) || (ii==ny-1) || (jj==0) || (jj==nx-1))
            {
                for (int kk=-1; kk<=1; ++kk)
                {
                    for (int ll=-1; ll<=1; ++ll)
                    {
                    if (!((kk==0) && (ll==0)))
                        neighbours(ii,jj) += board((ii+kk)%ny,(jj+ll)%nx);
                    }
                }
            } 
            else 
            {
               for (int kk=-1; kk<=1; ++kk)
                {
                    for (int ll=-1; ll<=1; ++ll)
                    {
                        if (!((kk==0) && (ll==0)))
                            neighbours(ii,jj) += board((ii+kk),(jj+ll));
                    }
                }
            }  
        }
    }
    """
    weave.inline(code,['neighbours', 'board', 'nx', 'ny'],
                 type_converters = converters.blitz,
                 compiler = 'gcc')
    return neighbours

def compute_neighbours_original(Z):
    rows, cols = len(Z), len(Z[0])
    N = np.zeros(np.shape(Z))
    for x in range(rows):
        for y in range(cols):
            for ii in [-1,0,1]:
                for jj in [-1,0,1]:
                    if ii==0 and jj==0:
                        continue
                    else:
                        N[x][y] += Z[(x+ii)%cols][(x+jj)%rows]
    return N

@jit
def compute_neighbours_original_jit(Z):
    rows, cols = len(Z), len(Z[0])
    N = np.zeros(np.shape(Z))
    for x in range(rows):
        for y in range(cols):
            for ii in [-1,0,1]:
                for jj in [-1,0,1]:
                    if ii==0 and jj==0:
                        continue
                    else:
                        N[x][y] += Z[(x+ii)%cols][(x+jj)%rows]
    return N

@profile
def iterate_original(Z):
    Zprime = Z.copy()
    rows, cols = len(Zprime), len(Zprime[0])
    N = compute_neighbours_original(Zprime)
    for x in range(rows):
        for y in range(cols):
            if Zprime[x][y] == 1:
                if (N[x][y] < 2) or (N[x][y] > 3):
                    Zprime[x][y] = 0
            else:
                if (N[x][y] == 3):
                    Zprime[x][y] = 1

    return Zprime

@jit
def iterate_original_jit(Z):
    Zprime = Z.copy()
    rows, cols = len(Zprime), len(Zprime[0])
    N = compute_neighbours_original_jit(Zprime)
    for x in range(rows):
        for y in range(cols):
            if Zprime[x][y] == 1:
                if (N[x][y] < 2) or (N[x][y] > 3):
                    Zprime[x][y] = 0
            else:
                if (N[x][y] == 3):
                    Zprime[x][y] = 1

    return Zprime

@profile
def iterate_original_jit_prof(Z):
    return iterate_original_jit(Z)

@profile
def iterate(board):
    neighbours = find_neighbours(board)
    set_zero_idxs = (board==1) & ((neighbours<2) | (neighbours>3))
    set_one_idxs = (board!=1) & (neighbours==3)
    board[set_zero_idxs] = 0
    board[set_one_idxs] = 1
    return board

@profile
def iterate_weave(board):
    neighbours = find_neighbours_weave(board)
    set_zero_idxs = (board==1) & ((neighbours<2) | (neighbours>3))
    set_one_idxs = (board!=1) & (neighbours==3)
    board[set_zero_idxs] = 0
    board[set_one_idxs] = 1
    return board

@profile
def iterate_jit(board):
    neighbours = find_neighbours_jit(board)
    set_zero_idxs = (board==1) & ((neighbours<2) | (neighbours>3))
    set_one_idxs = (board!=1) & (neighbours==3)
    board[set_zero_idxs] = 0
    board[set_one_idxs] = 1
    return board
    
def run(x,y,pattern_name,iterations,iterate_func=iterate):
    if pattern_name is not None:
        assert pattern_name in PATTERNS, "Valid pattern names are {}".format(PATTERNS.keys())
        board = init((x,y),PATTERNS[pattern_name])
    else:
        board = init((x,y))

    find_neighbours_weave(board)

    images = []
    for _ in range(iterations):
        images.append([plt.imshow(board, interpolation = 'nearest', cmap='binary')])
        board = iterate_func(board)
    images.append([plt.imshow(board, interpolation = 'nearest', cmap='binary')])
    return images

def animate(images,interval):
    fig = plt.gcf()
    ani = animation.ArtistAnimation(fig, images, interval=interval, blit=True, repeat=False)
    plt.show()

def precompile_funcs():
    board = np.zeros([10,10],dtype='int32')
    find_neighbours_weave(board)
    find_neighbours_jit(board)
    compute_neighbours_original_jit(board)
    iterate_original_jit(board)

if __name__ == "__main__":
    from argparse import ArgumentParser
    import sys
    parser = ArgumentParser(usage="python {} [args]".format(sys.argv[0]))
    parser.add_argument("-x","--ncols",help="The number of colums on the board",type=int,default=100)
    parser.add_argument("-y","--nrows",help="The number of rows on the board",type=int,default=100)
    parser.add_argument("-i","--nits",help="The number of iterations of the game",type=int,default=100)
    parser.add_argument("-p","--pattern",help="A pattern to inject onto the board [Default is random pattern]",type=str,default=None)
    parser.add_argument("-d","--delay",help="Interval between frames in the animation [lower is faster]",type=int,default=50)
    parser.add_argument("-s","--display",help="Display animation",action="store_true")
    args = parser.parse_args()
    precompile_funcs()
    images = run(args.nrows, args.ncols, args.pattern, args.nits, iterate_original_jit_prof)
    images = run(args.nrows, args.ncols, args.pattern, args.nits, iterate_original)
    images = run(args.nrows, args.ncols, args.pattern, args.nits, iterate_jit)
    images = run(args.nrows, args.ncols, args.pattern, args.nits, iterate)
    images = run(args.nrows, args.ncols, args.pattern, args.nits, iterate_weave)
    if args.display:
        animate(images, args.delay)

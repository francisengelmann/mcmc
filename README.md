## Markov Chain Monte Carlo - Metropolis Hastings

This program shows a compact implementation of the MCMC-MH algorithm
for density estimation and solving a Sudoku puzzle.

### Compilation

OpenCV is required for visualizations.

Run these commands from the root directory of the project:
```
$ mkdir build
$ cd build
$ cmake ..
$ make install
```

To run the executables type:
```
$ ./bin/mcmc
```
or
```
$ ./bin/sudoku
```

3. REFERENCES
-------------
[1] D. MacKay. Information Theory, Inference, and Learning Algorithms.
    http://www.inference.eng.cam.ac.uk/mackay/itila/book.html

[2] E. Chi, and K. Lange. Techniques for Solving Sudoku Puzzles.
    http://arxiv.org/abs/1203.2295
    
[3] J. Goldstick. Statistics 406: Introduction to Statistical Computing.
    http://dept.stat.lsa.umich.edu/~jasoneg/Stat406/lab5.pdf

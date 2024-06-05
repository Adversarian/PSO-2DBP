# PSO-2DBP
Solving a specific variant of the of two-dimensional bin-packing (2DBP) with Particle Swarm Optimization (PSO)

- This algorithm initializes the first particle of the swarm with the Best Fit Decreasing(BFD) heuristic and assigns a higher social coefficient to guide solutions towards the guaranteed feasible solution produced by BFD.
- 90 degree rotations of the items are allowed and are implemented by swapping the height and width of an item.
- The position of each particle is represented by an array of length `num_boxes`(number of items) of 4-tuples `(bin_number, x, y, rotated)`, where `bin_number` is the id of the bin in which the box is placed, `x, y` denote the coordinates of the bottom-left corner of the box which uniquely identify the box as its height and width are also stored as class attributes(but do not consitute the position of the particle) and finally `rotated` is a binary variable that shows if a box is rotated or not.

# Usage
```python
python main.py -tc test_case.txt -mi 100 -np 100 -t 10 -v -s
```

# Todo
- Further testing.
- Docs? Maybe? Idk.
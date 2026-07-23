# `test_density_rule.py`

These tests make the learned H7 component's restrictions executable. Sigmoid
coefficients remain nonnegative and below the frozen maxima under extreme
inputs; zero-logit initialization produces the declared midpoint force law;
and output acceleration follows only the supplied explicit vector directions
while respecting the smooth 12 m/s² norm cap.

# `test_h7d_experiment.py`

This test fixes H7D's verdict boundary: an ecological clip may omit only the
three gates requiring materially excited density behavior. Teacher and rollout
safety/accuracy gates remain mandatory, and any such failure rejects the arm.

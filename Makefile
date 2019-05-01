# running a Python test
test:
	PYTHONPATH=. coverage run tests/test_cardiogram_experiment.py
	PYTHONPATH=. coverage run tests/test_gabor_experiment.py
	PYTHONPATH=. coverage run tests/test_shapes_experiment.py

figures:
	python utils/figures.py

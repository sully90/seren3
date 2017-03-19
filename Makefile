
FRIEDMANN_DIR=utils/f90/

.PHONY all:
	make cython
	make fortran
	make inplace_build

# This compiles all modules for an inplace use
inplace_build:
	python setup.py build_src build_ext --inplace $(PYSETUP_FLAGS)

cython:
	find . -name "*.pyx" | xargs cython

fortran:
	cd $(FRIEDMANN_DIR) && f2py -m friedmann -c friedmann.f90

clean:
	find . -type f -name "*.pyc" -exec rm {} \;
	find . -type f -name "*.so" -exec rm {} \;
	rm -rf build
# 1. Path to your virtual environment's python
# This ensures we use your .venv even if it isn't "activated" in the terminal
PYTHON = $(shell pwd)/.venv/bin/python3
THREADS = $(shell nproc)

all:
# Only configure if the build directory doesn't exist
	@if [ ! -d "build" ]; then \
		cmake -B build -S . -DCMAKE_BUILD_TYPE=Release -DPython3_EXECUTABLE=$(PYTHON); \
	fi
# Use -j to use all CPU cores
	cmake --build build -j $(shell nproc)
	cp build/stochastic_flock*.so stochastic_flock

native:
	cmake -B build_native -S . -DCMAKE_BUILD_TYPE=Release -DNATIVE_OPTIM=ON -DPython3_EXECUTABLE=$(PYTHON)
	cmake --build build_native -j $(THREADS)

pgo:
# STEP 1: Generate profiles
	cmake -B build_pgo -S . -DCMAKE_BUILD_TYPE=Release -DNATIVE_OPTIM=ON -DCMAKE_CXX_FLAGS="-fprofile-generate" -DPython3_EXECUTABLE=$(PYTHON)
	cmake --build build_pgo  -j $(THREADS)
	cp build_pgo/stochastic_flock*.so stochastic_flock
	./build_pgo/bird_solver
# Run using the VENV python
	$(PYTHON) tuning.py
# STEP 2: Use profiles
	find build_pgo -name "*.gcda"

	cmake -B build_pgo -S . -DCMAKE_BUILD_TYPE=Release -DNATIVE_OPTIM=ON -DCMAKE_CXX_FLAGS="-fprofile-use" -DPython3_EXECUTABLE=$(PYTHON)
	cmake --build build_pgo  -j $(THREADS)
	find build_pgo -name "*.gcda" -delete

debug:
	cmake -B build_debug -S . -DCMAKE_BUILD_TYPE=Debug -DCMAKE_CXX_FLAGS="-fsanitize=address,undefined" -DPython3_EXECUTABLE=$(PYTHON)
	cmake --build build_debug  -j $(THREADS)

clean:
	rm -rf build build_native build_debug build_pgo *.so *.gcda dist

help:
	@echo "all:    Standard build"
	@echo "native: Fast build with native hardware optimizations"
	@echo "pgo:    Profile guided optimization + native build, max performance (requires tuning.py)"
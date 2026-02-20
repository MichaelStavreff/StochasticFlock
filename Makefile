# PYTHON = $(shell pwd)/.venv/bin/python3
# THREADS = $(shell nproc)

# init:
# 	$(PYTHON) -m pip install -e .
# all:
# # Only configure if the build directory doesn't exist
# 	@if [ ! -d "build" ]; then \
# 		cmake -B build -S . -DCMAKE_BUILD_TYPE=Release -DPython3_EXECUTABLE=$(PYTHON); \
# 	fi
	
# 	cmake --build build -j $(THREADS)
# 	cp build/stochastic_flock*.so src/stochastic_flock

# native:
# 	cmake -B build_native -S . -DCMAKE_BUILD_TYPE=Release -DNATIVE_OPTIM=ON -DPython3_EXECUTABLE=$(PYTHON)
# 	cmake --build build_native -j $(THREADS)

# pgo: #only does bird_solver PGO

# 	cmake -B build_pgo -S . -DCMAKE_BUILD_TYPE=Release -DNATIVE_OPTIM=ON -DCMAKE_CXX_FLAGS="-fprofile-generate" -DPython3_EXECUTABLE=$(PYTHON)
# 	cmake --build build_pgo --target bird_solver -j $(THREADS)

# 	./build_pgo/bird_solver
	
# 	cmake -B build_pgo -S . -DCMAKE_BUILD_TYPE=Release -DNATIVE_OPTIM=ON -DCMAKE_CXX_FLAGS="-fprofile-use" -DPython3_EXECUTABLE=$(PYTHON)
# 	cmake --build build_pgo --target bird_solver  -j $(THREADS)

# debug:
# 	cmake -B build_debug -S . -DCMAKE_BUILD_TYPE=Debug -DCMAKE_CXX_FLAGS="-fsanitize=address,undefined" -DPython3_EXECUTABLE=$(PYTHON)
# 	cmake --build build_debug  -j $(THREADS)

# clean:
# 	rm -rf build build_native build_debug build_pgo dist
# 	find . -name "*.gcda" -delete
# 	find . -name "*.so" -delete

# help:
# 	@echo "all:    Standard build"
# 	@echo "native: Fast build with native hardware optimizations"
# 	@echo "pgo:    Profile guided optimization + native build, max performance"


PYTHON = $(shell pwd)/.venv/bin/python3
THREADS = $(shell nproc)
.PHONY: dev portable native pgo clean help

help:
	@echo "========================================================================"
	@echo "make dev       : FASTEST build. Debug mode, no LTO, executable only."
	@echo "                 "
	@echo "make portable  : PROD build. Optimized (-O3), LTO on, but PORTABLE."
	@echo "                 Builds Python module and executable."
	@echo
	@echo "make native    : MAX SPEED build. Optimized for native architecture, not portable."
	@echo "                "
	@echo "make pgo       : EXTREME build. Uses profile-guided optimization."
	@echo "make clean     : Remove all build directories and compiled binaries."
	@echo "========================================================================"

# 1. DEV: Just the executable, no optimizations, no LTO, instant builds
dev:
	@cmake -B build_dev -S . -DCMAKE_BUILD_TYPE=Debug -DPython3_EXECUTABLE=$(PYTHON)
	@cmake --build build_dev --target bird_solver -j $(THREADS)
	@echo "Dev build complete: ./build_dev/bird_solver"

# 2. PORTABLE: Python & Executable at max speed, but works on all CPUs (No -march=native)
portable:
	@cmake -B build_portable -S . -DCMAKE_BUILD_TYPE=Release -DNATIVE_OPTIM=OFF -DPython3_EXECUTABLE=$(PYTHON)
	@cmake --build build_portable -j $(THREADS)
	@cp build_portable/stochastic_flock*.so src/stochastic_flock/ 2>/dev/null || true
	@echo "Portable build complete in build_portable/"

# 3. NATIVE: Max speed for YOUR specific CPU (not portable)
native:
	@cmake -B build_native -S . -DCMAKE_BUILD_TYPE=Release -DNATIVE_OPTIM=ON -DPython3_EXECUTABLE=$(PYTHON)
	@cmake --build build_native -j $(THREADS)

# 4. PGO: Profile Guided Optimization (Max possible performance)
pgo:
	@cmake -B build_pgo -S . -DCMAKE_BUILD_TYPE=Release -DNATIVE_OPTIM=ON -DCMAKE_CXX_FLAGS="-fprofile-generate" -DPython3_EXECUTABLE=$(PYTHON)
	@cmake --build build_pgo --target bird_solver -j $(THREADS)
	@./build_pgo/bird_solver --benchmark # Assume you add a benchmark flag
	@cmake -B build_pgo -S . -DCMAKE_BUILD_TYPE=Release -DNATIVE_OPTIM=ON -DCMAKE_CXX_FLAGS="-fprofile-use" -DPython3_EXECUTABLE=$(PYTHON)
	@cmake --build build_pgo -j $(THREADS)

clean:
	rm -rf build_dev build build_portable build_native build_pgo src/stochastic_flock/*.so


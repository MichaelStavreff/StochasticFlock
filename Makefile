PYTHON = $(shell pwd)/.venv/bin/python3
THREADS = $(shell nproc)

init:
	$(PYTHON) -m pip install -e .
all:
# Only configure if the build directory doesn't exist
	@if [ ! -d "build" ]; then \
		cmake -B build -S . -DCMAKE_BUILD_TYPE=Release -DPython3_EXECUTABLE=$(PYTHON); \
	fi
	
	cmake --build build -j $(THREADS)
	cp build/stochastic_flock*.so src/stochastic_flock

native:
	cmake -B build_native -S . -DCMAKE_BUILD_TYPE=Release -DNATIVE_OPTIM=ON -DPython3_EXECUTABLE=$(PYTHON)
	cmake --build build_native -j $(THREADS)

pgo: #only does bird_solver PGO

	cmake -B build_pgo -S . -DCMAKE_BUILD_TYPE=Release -DNATIVE_OPTIM=ON -DCMAKE_CXX_FLAGS="-fprofile-generate" -DPython3_EXECUTABLE=$(PYTHON)
	cmake --build build_pgo --target bird_solver -j $(THREADS)

	./build_pgo/bird_solver
	
	cmake -B build_pgo -S . -DCMAKE_BUILD_TYPE=Release -DNATIVE_OPTIM=ON -DCMAKE_CXX_FLAGS="-fprofile-use" -DPython3_EXECUTABLE=$(PYTHON)
	cmake --build build_pgo --target bird_solver  -j $(THREADS)

debug:
	cmake -B build_debug -S . -DCMAKE_BUILD_TYPE=Debug -DCMAKE_CXX_FLAGS="-fsanitize=address,undefined" -DPython3_EXECUTABLE=$(PYTHON)
	cmake --build build_debug  -j $(THREADS)

clean:
	rm -rf build build_native build_debug build_pgo *.so *.gcda dist src/stochastic_flock/*.so

help:
	@echo "all:    Standard build"
	@echo "native: Fast build with native hardware optimizations"
	@echo "pgo:    Profile guided optimization + native build, max performance"

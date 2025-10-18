.PHONY: all build test clean

all: build

build:
	@uv sync --reinstall-package cynn

test:
	@uv run pytest -v

cmake:
	@mkdir -p build && cd build && cmake .. && cmake --build . --config Release

clean:
	@rm -rf build src/*.egg-info
	@rm -f *.so

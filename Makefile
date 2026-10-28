# Reproducibility entry points for the source-only repository.
#
# The canonical contract remains `python -m explib.<driver> <mode>` (see README); this
# file only groups the zero-GPU checks so a fresh checkout can be verified with
# one command and CI/contributors do not have to rediscover the order.
PYTHON ?= python
BUILD_DIR := .build
INSTALL_DIR := $(BUILD_DIR)/install-check

.PHONY: help test plan audit install-check build clean

help:
	@echo "make test          # CPU unit/reproducibility suite (zero GPU)"
	@echo "make plan          # frozen-manifest preflight, no test-split read"
	@echo "make audit         # local audit gates -> audit_report.json"
	@echo "make install-check # verify the project builds/installs as a wheel"
	@echo "make build         # sdist + wheel via python -m build"
	@echo "make clean         # remove packaging scratch space"

test:
	$(PYTHON) test_review_fixes.py

plan:
	$(PYTHON) -m explib.confirmation plan

audit:
	$(PYTHON) audit_gates.py report

install-check:
	rm -rf "$(INSTALL_DIR)"
	mkdir -p "$(INSTALL_DIR)"
	$(PYTHON) -m pip install --no-deps --no-build-isolation \
		--target "$(INSTALL_DIR)" .

build:
	$(PYTHON) -m build --sdist --wheel --outdir "$(BUILD_DIR)/dist"

clean:
	rm -rf "$(BUILD_DIR)" *.egg-info

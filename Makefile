# Reproducibility entry points for the source-only repository.
#
# The canonical contract remains `python -m explib.<driver> <mode>` (see README); this
# file only groups the zero-GPU checks so a fresh checkout can be verified with
# one command and CI/contributors do not have to rediscover the order.
PYTHON ?= python
BUILD_DIR := .build
INSTALL_DIR := $(BUILD_DIR)/install-check

.PHONY: help test plan protocol preparation audit manipulation integrity env-check registry data-sensitivity reproduce report-diff install-check build clean

help:
	@echo "make test          # CPU unit/reproducibility suite (zero GPU)"
	@echo "make plan          # frozen-manifest preflight, no test-split read"
	@echo "make protocol      # prospective design registration gate (design_protocol.json)"
	@echo "make preparation   # launch clearance: protocol + allocation + power + rehearsal (zero GPU)"
	@echo "make audit         # local audit gates -> audit_report.json (non-zero on a failed gate)"
	@echo "make manipulation  # functional manipulation check for every declared ablation (zero GPU)"
	@echo "make integrity     # verify stamped analysis bodies/inputs (zero GPU)"
	@echo "make env-check     # local lock/OS stack vs recorded env_fp_v2 (zero GPU)"
	@echo "make registry      # split-package _PARTS registration integrity (zero GPU)"
	@echo "make data-sensitivity # cleaning-choice/order sensitivity of the cached corpus (zero GPU)"
	@echo "make reproduce A=<dir> B=<dir>  # compare two independent runs (zero GPU)"
	@echo "make report-diff A=<json> B=<json>  # compare two stamped reports (zero GPU)"
	@echo "make install-check # verify the project builds/installs as a wheel"
	@echo "make build         # sdist + wheel via python -m build"
	@echo "make clean         # remove packaging scratch space"

test:
	$(PYTHON) test_review_fixes.py

plan:
	$(PYTHON) -m explib.confirmation plan

protocol:
	$(PYTHON) audit_gates.py design_protocol

preparation:
	$(PYTHON) audit_gates.py preparation_clearance

audit:
	$(PYTHON) audit_gates.py report

manipulation:
	$(PYTHON) audit_gates.py manipulation

integrity:
	$(PYTHON) audit_gates.py analysis_integrity

env-check:
	$(PYTHON) audit_gates.py environment_identity

registry:
	$(PYTHON) audit_gates.py split_registry

data-sensitivity:
	$(PYTHON) audit_gates.py cleaning_sensitivity

reproduce:
	@test -n "$(A)" -a -n "$(B)" || { \
		echo "usage: make reproduce A=<original result dir> B=<rerun result dir>"; \
		exit 2; }
	$(PYTHON) -m explib.reproduce compare "$(A)" "$(B)"

report-diff:
	@test -n "$(A)" -a -n "$(B)" || { \
		echo "usage: make report-diff A=<old report.json> B=<rebuilt report.json>"; \
		exit 2; }
	$(PYTHON) -m explib.reproduce report "$(A)" "$(B)" --ignore-ledger

install-check:
	rm -rf "$(INSTALL_DIR)"
	mkdir -p "$(INSTALL_DIR)"
	$(PYTHON) -m pip install --no-deps --no-build-isolation \
		--target "$(INSTALL_DIR)" .

build:
	$(PYTHON) -m build --sdist --wheel --outdir "$(BUILD_DIR)/dist"

clean:
	rm -rf "$(BUILD_DIR)" *.egg-info

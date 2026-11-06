# Reproducibility entry points for the source-only repository.
#
# The canonical contract remains `python -m explib.<driver> <mode>` (see README); this
# file only groups the zero-GPU checks so a fresh checkout can be verified with
# one command and CI/contributors do not have to rediscover the order.
PYTHON ?= python
BUILD_DIR := .build
INSTALL_DIR := $(BUILD_DIR)/install-check

.PHONY: help test plan protocol preparation rehearsal transfer budget attribution ablation-rank ablation-screen sink-interaction sequential-design budget-ablation curve-inference exact-signflip-ci estimability conformal type-sm exact-tost global-fdr multiverse audit manipulation integrity env-check canary canary-check registry data-sensitivity reproduce report-diff install-check build clean

help:
	@echo "make test          # CPU unit/reproducibility suite (zero GPU)"
	@echo "make plan          # frozen-manifest preflight, no test-split read"
	@echo "make protocol      # prospective design registration gate (design_protocol.json)"
	@echo "make preparation   # launch clearance: protocol + allocation + power + rehearsal (zero GPU)"
	@echo "make rehearsal     # known-truth rehearsal of the frozen decision path (zero GPU)"
	@echo "make transfer      # validation -> test effect transfer / shrinkage (zero GPU)"
	@echo "make budget        # budget-truncation feasibility: cells, retries, completion risk (zero GPU)"
	@echo "make attribution   # ablation lattice: Shapley/interaction design licensing + replay (zero GPU)"
	@echo "make ablation-rank # joint Shapley ranking: max-T intervals, pair order, scale sensitivity (zero GPU)"
	@echo "make ablation-screen # budgeted fractional-factorial ablation screening: aliasing + replay (zero GPU)"
	@echo "make sink-interaction # 2x2 difference-in-differences: exact interaction + family max-T (zero GPU)"
	@echo "make sequential-design # prospective group-sequential plan: exact alpha spending (zero GPU)"
	@echo "make budget-ablation # budget x ablation interaction: paired log-PPL slopes + exact TOST (zero GPU)"
	@echo "make curve-inference # curve-level simultaneous band, flatness and uniform IUT claims (zero GPU)"
	@echo "make exact-signflip-ci # exact seed-level confidence set consistent with the sign-flip p (zero GPU)"
	@echo "make estimability  # design matrix rank / contrast estimability (zero GPU)"
	@echo "make conformal    # distribution-free prediction interval replay (zero GPU)"
	@echo "make type-sm       # design-stage Type S / Type M sign & magnitude risk (zero GPU)"
	@echo "make exact-tost    # distribution-free exact sign-flip TOST companion (zero GPU)"
	@echo "make global-fdr    # cross-panel exploratory BH/BY false-discovery-rate replay (zero GPU)"
	@echo "make multiverse    # declared analysis multiverse: IUT robustness + within-family max-T (zero GPU)"
	@echo "make audit         # local audit gates -> audit_report.json (non-zero on a failed gate)"
	@echo "make manipulation  # functional manipulation check for every declared ablation (zero GPU)"
	@echo "make integrity     # verify stamped analysis bodies/inputs (zero GPU)"
	@echo "make env-check     # local lock/OS stack vs recorded env_fp_v2 (zero GPU)"
	@echo "make canary        # record this process's arithmetic canary (zero GPU)"
	@echo "make canary-check  # replay analysis_v12/numerical_canary.json (zero GPU)"
	@echo "make registry      # split-package _PARTS registration integrity (zero GPU)"
	@echo "make data-sensitivity # cleaning-choice/order sensitivity of the cached corpus (zero GPU)"
	@echo "make reproduce A=<dir> B=<dir>  # compare source/data/init/endpoint/curve (zero GPU)"
	@echo "make report-diff A=<json> B=<json>  # compare two stamped reports (zero GPU)"
	@echo "make install-check # verify the project builds/installs as a wheel"
	@echo "make build         # sdist + wheel via python -m build"
	@echo "make clean         # remove packaging scratch space"
	@echo ""
	@echo "reproducibility targets:"
	@echo "make part-hash       # print the _PARTS list SHA-256 hash"

test:
	$(PYTHON) test_review_fixes.py

plan:
	$(PYTHON) -m explib.confirmation plan

protocol:
	$(PYTHON) audit_gates.py design_protocol

preparation:
	$(PYTHON) audit_gates.py preparation_clearance

rehearsal:
	$(PYTHON) audit_gates.py analysis_rehearsal

transfer:
	$(PYTHON) audit_gates.py validation_transfer

budget:
	$(PYTHON) audit_gates.py budget_truncation

attribution:
	$(PYTHON) audit_gates.py ablation_lattice

ablation-rank:
	$(PYTHON) audit_gates.py ablation_ranking

ablation-screen:
	$(PYTHON) audit_gates.py ablation_screening

estimability:
	$(PYTHON) audit_gates.py design_estimability

conformal:
	$(PYTHON) audit_gates.py distribution_free_prediction

type-sm:
	$(PYTHON) audit_gates.py type_sm

exact-tost:
	$(PYTHON) audit_gates.py exact_tost

global-fdr:
	$(PYTHON) audit_gates.py global_fdr

multiverse:
	$(PYTHON) audit_gates.py analysis_multiverse

sink-interaction:
	$(PYTHON) audit_gates.py contrast_interaction

sequential-design:
	$(PYTHON) audit_gates.py sequential_design

budget-ablation:
	$(PYTHON) audit_gates.py budget_ablation

curve-inference:
	$(PYTHON) audit_gates.py curve_inference

exact-signflip-ci:
	$(PYTHON) audit_gates.py exact_signflip_ci

audit:
	$(PYTHON) audit_gates.py report

manipulation:
	$(PYTHON) audit_gates.py manipulation

integrity:
	$(PYTHON) audit_gates.py analysis_integrity

env-check:
	$(PYTHON) audit_gates.py environment_identity

canary:
	$(PYTHON) -m explib.reproduce canary --write

canary-check:
	$(PYTHON) audit_gates.py numerical_canary

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

part-hash:
	@$(PYTHON) -c "from explib import part_list_hash; print(part_list_hash())"

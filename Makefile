# ENDGAME — Top-level Makefile
# Builds the FDq Fortran differentiation-matrix binary.
# Usage:
#   make          — build FDq
#   make clean    — remove build artefacts
#   make rebuild  — clean then build

.PHONY: all clean rebuild

all:
	$(MAKE) -C FDq/bin

clean:
	$(MAKE) -C FDq/bin clean

rebuild: clean all

.ONESHELL:
SHELL = /bin/bash


# Create or update the environment from environment.yml with no activation
.PHONY: env
env:
	source /srv/conda/etc/profile.d/conda.sh
	conda env update -n "myst-proj03" -f "environment.yml" --prune \
	|| conda env create -n "myst-proj03" -f "environment.yml"


# Run all notebooks using nbconvert
.PHONY: all
all: env
	source /srv/conda/etc/profile.d/conda.sh
	conda activate "myst-proj03"
	for nb in *.ipynb; do \
		[ -e "$$nb" ] || continue; \
		jupyter nbconvert --to notebook --execute "$$nb" --inplace; \
	done


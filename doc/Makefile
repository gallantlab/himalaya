# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line.
SPHINXOPTS    =
SPHINXBUILD   = sphinx-build
SPHINXPROJ    = Tutorials
SOURCEDIR     = .
BUILDDIR      = _build

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

clean:
	rm -rf $(BUILDDIR)/*
	rm -rf _auto_examples/
	rm -rf _generated/

html-noplot:
	$(SPHINXBUILD) -D plot_gallery=0 -b html "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
	@echo
	@echo "Build finished (noplot). The HTML pages are in $(BUILDDIR)/html."

# -b gh_pages --single-branch (to clone only one branch)
# --no-checkout (just fetches the root folder without content)
# --depth 1 (since we don't need the history prior to the last commit)
push-pages:
	rm -rf _build/gh_pages
	git clone -b gh-pages --single-branch --no-checkout --depth 1 \
		https://github.com/gallantlab/himalaya _build/gh_pages

	cd _build/ && \
		cp -r html/* gh_pages && \
		cd gh_pages && \
		touch .nojekyll && \
		git add * && \
		git add .nojekyll && \
		git commit -a -m 'Make push-pages' && \
		git push

# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line.
SPHINXOPTS    =
SPHINXBUILD   = sphinx-build
SPHINXPROJ    = GluonTS
SOURCEDIR     = .
BUILDDIR      = _build


APIDOC        = sphinx-apidoc
APIDOC_OPTS   = --implicit-namespaces --separate --module-first
APIDOC_ROOT   = gluonts

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile

apidoc:
	@rm -Rf api/$(APIDOC_ROOT)
	@$(APIDOC) $(APIDOC_OPTS) -o api/$(APIDOC_ROOT) ../src/$(APIDOC_ROOT) setup* test docs *pycache*
	@rm -Rf api/$(APIDOC_ROOT)/modules.rst
	@sed -i"" -e "s/$(APIDOC_ROOT) package/API Docs/" api/$(APIDOC_ROOT)/$(APIDOC_ROOT).rst

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: apidoc Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

# appended
import sys
import os
import sphinx_rtd_theme


# Add project root to PATH
sys.path.append(os.path.abspath(".."))

extensions.append("sphinx_rtd_theme")

html_theme = "sphinx_rtd_theme"

# Fix ReadTheDocs build error
master_doc = "index"

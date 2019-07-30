sphinx-quickstart \
    --sep \
    -p Babysister \
    -a "Tran Le Thai Son" \
    -v 2.0.0 \
    --ext-autodoc \
    --ext-intersphinx \
    --ext-imgmath \
    --ext-githubpages \
    --extensions sphinx.ext.napoleon \
    ./

CONF="./source/conf.py "
ROOT="../.."
printf "import sys\nimport os\n\nsys.path.append(os.path.abspath(\"$ROOT\"))\n\n" \
    | cat - $CONF > temp
mv temp $CONF

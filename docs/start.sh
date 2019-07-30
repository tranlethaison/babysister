sphinx-quickstart \
    -p Babysister \
    -a "Tran Le Thai Son" \
    --ext-autodoc \
    --ext-intersphinx \
    --ext-imgmath \
    --ext-githubpages \
    --extensions sphinx.ext.napoleon \
    ./

CONF="./conf.py"
ROOT=".."
printf "import sys\nimport os\n\nsys.path.append(os.path.abspath('$ROOT'))\n\n" \
    | cat - $CONF > temp
mv temp $CONF

printf "\n# Fix ReadTheDocs build error\nmaster_doc = 'index'" >> $CONF

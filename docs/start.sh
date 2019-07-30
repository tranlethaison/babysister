printf "> Clean docs\n"
rm -rf *.rst _* Makefile make.bat conf.py

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
APPEND="./append_to_conf.py"
printf "> Append $APPEND to $CONF\n"
cat $APPEND >> $CONF

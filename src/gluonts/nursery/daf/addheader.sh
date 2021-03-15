for i in `find . -name "*.py" -type f`;
do
    cat copyright $i > $i.new && mv $i.new $i
done
    
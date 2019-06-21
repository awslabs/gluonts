import sys
import time
import notedown
import nbformat

assert len(sys.argv) == 2, 'usage: input.md'

# timeout for each notebook, in sec
timeout = 40 * 60

# the files will be ignored for execution
ignore_execution = []

input_fn = sys.argv[1]
output_fn = '.'.join(input_fn.split('.')[:-1] + ['ipynb'])

reader = notedown.MarkdownReader()

# read
with open(input_fn, 'r') as f:
    notebook = reader.read(f)

if not any([i in input_fn for i in ignore_execution]):
    tic = time.time()
    notedown.run(notebook, timeout)
    print('=== Finished evaluation in %f sec' % (time.time() - tic))

# write
# need to add language info to for syntax highlight
notebook['metadata'].update({'language_info': {'name': 'python'}})

with open(output_fn, 'w') as f:
    f.write(nbformat.writes(notebook))

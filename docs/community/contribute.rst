Contribute
==========

GluonTS community welcomes contributions from anyone! Latest documentation can be found `here <https://ts.gluon.ai/master/index.html>`__.

There are lots of opportunities for you to become our `contributors <https://github.com/awslabs/gluon-ts/blob/master/contributor.rst>`__:

- Ask or answer questions on `GitHub issues <https://github.com/awslabs/gluon-ts/issues>`__.
- Propose ideas, or review proposed design ideas on `GitHub issues <https://github.com/awslabs/gluon-ts/issues>`__.
- Improve the `documentation <https://ts.gluon.ai/master/index.html>`__.
- Contribute bug reports `GitHub issues <https://github.com/awslabs/gluon-ts/issues>`__.
- Write new `tutorials <https://github.com/awslabs/gluon-ts/tree/master/docs/tutorials>`__.
- Most importantly, if you have an idea of how to contribute, then do it!

For a list of open starter tasks, check `good first issues <https://github.com/awslabs/gluon-ts/labels/good%20first%20issue>`__.

- `Make changes <#make-changes>`__

- `Contribute tutorials <#contribute-tutorials>`__

- `Contribute new Models <#contribute-new-model>`__

- `Git Workflow Howtos <#git-workflow-howtos>`__

   -  `How to submit pull request <#how-to-submit-pull-request>`__
   -  `How to resolve conflict with
      master <#how-to-resolve-conflict-with-master>`__
   -  `How to combine multiple commits into
      one <#how-to-combine-multiple-commits-into-one>`__
   -  `What is the consequence of force
      push <#what-is-the-consequence-of-force-push>`__


Make changes
------------

Our package uses continuous integration and code coverage tools for verifying pull requests. Before
submitting, contributor should perform the following checks:

- `Lint (code style) check <https://github.com/awslabs/gluon-ts/blob/master/ci/jenkins/build_steps.groovy>`__.
- `Py3 <https://github.com/awslabs/gluon-ts/blob/master/ci/jenkins/Jenkinsfile_py3_cpu_unittest>`__ tests.


Contribute tutorials
--------------------

Our :doc:`tutorials <../tutorials/index>` are intended for people who
are interested in time series and want to get better familiarized on different parts in time series. In order for
people to easily understand the content, the code needs to be clean and readable, accompanied by
explanation with good writing.

See `existing tutorials <https://github.com/awslabs/gluon-ts/tree/master/docs/tutorials>`__.

To make the review process easy, we adopt `notedown <https://github.com/aaren/notedown>`_ as the
tutorial format. Notedown notebooks are regular markdown files with code blocks that can be
converted into `Jupyter notebooks <http://jupyter.org/>`_.

We suggest you start the example with `Jupyter notebook <http://jupyter.org/>`_. When the content is ready, please:

- Clear the output cells in the jupyter notebook,
- `Install notedown <https://github.com/aaren/notedown>`_.
- Run `notedown input.ipynb --to markdown > output.md`
- Submit the `.md` file for review.

Notebook Guidelines:

- Less is better. Only show the code that needs people's attention.
- Have a block upfront about the key takeaway of the notebook.
- Explain the motivation of the notebook to guide readers. Add figures if they help.
- Try to have < 10 lines of code per block, < 100 lines of code per notebook.
- Hide uninteresting complex functions in .py and import them.
- Hide uninteresting model parameters. We can make some of them default parameters in model definition. Maybe out of 30 we just show 5 interesting ones and pass those to model constructor.
- Only import module instead of classes and functions (i.e. from gluonts import model and use model.get_model, instead of from gluonts.model import get_model)
- Make tutorials more engaging, interactive, prepare practice questions for people to try it out. For example, for embedding evaluation, we can ask questions to the audience like what's the most similar word to xxx.
- Make sure the notebook can be zoomed in and still render well. This helps accommodate different viewing devices.
- For low level APIs such as BeamSearch and Scorer, explain the API with examples so ppl know how to play with it / hack it.


Contribute Docs
---------------

Documentation is at least as important as code. Good documentation delivers the correct message clearly and concisely.
If you see any issue in the existing documentation, a patch to fix is most welcome! To locate the
code responsible for the doc, you may use "Edit on Github" in the top right corner, or the
"[source]" links after each API. Also, `git grep` works nicely for searching for a specific string.

Git Workflow Howtos
-------------------

How to submit pull request
~~~~~~~~~~~~~~~~~~~~~~~~~~

-  Before submit, please rebase your code on the most recent version of
   master, you can do it by

.. code:: bash

    git remote add upstream https://github.com/awslabs/gluon-ts
    git fetch upstream
    git rebase upstream/master

-  If you have multiple small commits, it might be good to merge them
   together(use git rebase then squash) into more meaningful groups.
-  Send the pull request!

   -  Fix the problems reported by automatic checks
   -  If you are contributing a new module or new function, add a test.

How to resolve conflict with master
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

-  First rebase to most recent master

.. code:: bash

    # The first two steps can be skipped after you do it once.
    git remote add upstream https://github.com/awslabs/gluon-ts
    git fetch upstream
    git rebase upstream/master

-  The git may show some conflicts it cannot merge, say
   ``conflicted.py``.

   -  Manually modify the file to resolve the conflict.
   -  After you resolved the conflict, mark it as resolved by

   .. code:: bash

       git add conflicted.py

-  Then you can continue rebase by

.. code:: bash

    git rebase --continue

-  Finally push to your fork, you may need to force push here.

.. code:: bash

    git push --force

How to combine multiple commits into one
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Sometimes we want to combine multiple commits, especially when later
commits are only fixes to previous ones, to create a PR with set of
meaningful commits. You can do it by following steps. - Before doing so,
configure the default editor of git if you haven’t done so before.

.. code:: bash

    git config core.editor the-editor-you-like

-  Assume we want to merge last 3 commits, type the following commands

.. code:: bash

    git rebase -i HEAD~3

-  It will pop up an text editor. Set the first commit as ``pick``, and
   change later ones to ``squash``.
-  After you saved the file, it will pop up another text editor to ask
   you modify the combined commit message.
-  Push the changes to your fork, you need to force push.

.. code:: bash

    git push --force

Reset to the most recent master
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You can always use git reset to reset your version to the most recent
master. Note that all your ***local changes will get lost***. So only do
it when you do not have local changes or when your pull request just get
merged.

.. code:: bash

    git reset --hard [hash tag of master]
    git push --force

What is the consequence of force push
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The previous two tips requires force push, this is because we altered
the path of the commits. It is fine to force push to your own fork, as
long as the commits changed are only yours.

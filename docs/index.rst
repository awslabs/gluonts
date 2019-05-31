GluonTS - Probabilistic Time Series Modeling
============================================

Gluon Time Series (GluonTS) is the Gluon toolkit for probabilistic time series modeling, focusing on deep learning-based models.

GluonTS provides utilities for loading and iterating over time series datasets,
state of the art models ready to be trained, and building blocks to define
your own models and quickly experiment with different solutions. With GluonTS you can:

* Train and evaluate any of the built-in models on your own data, and quickly come up with a solution for your time series tasks.
* Use the provided abstractions and building blocks to create custom time series models, and rapidly benchmark them against baseline algorithms.


Get Started: A Quick Example
----------------------------

Here is a simple time series example with GluonTS for predicting Twitter volume with DeepAR.

(You can click the play button below to run this example.)

.. container:: demo
   :name: frontpage-demo

   `DeepAR for Twitter Volume <https://repl.it/@szha/gluon-ts>`_

.. raw:: html

   <script type="text/javascript">
   window.onload = function() {
     var demo = document.createElement("IFRAME");
     demo.src = "https://repl.it/@szha/gluon-ts?lite=true";
     demo.height = "600px";
     demo.width = "100%";
     demo.scrolling = "no";
     demo.frameborder = "no";
     demo.allowtransparency = true;
     demo.allowfullscreen = true;
     demo.seamless = true;
     demo.sandbox = "allow-forms allow-pointer-lock allow-same-origin allow-scripts allow-modals";
     demo_div = document.getElementById("frontpage-demo");
     while (demo_div.firstChild) {
       demo_div.removeChild(demo_div.firstChild);
     }
     demo_div.appendChild(demo);
   }; // load demo last
   </script>




Installation
------------

You can install GluonTS using `pip <https://pip.pypa.io/en/stable/installing/>`_ simply by running

.. code-block:: console

   pip install gluonts


.. hint::

   For more detailed guide on installing GluonTS,
   click the :doc:`install <install>` link in the top navigation bar.


Tutorials
---------
The best way to get started with GluonTS is by diving in using our :doc:`tutorials <examples/index>`, which you can download as Jupyter notebooks.


.. toctree::
   :hidden:
   :maxdepth: 2

   examples/index
   api/gluonts/gluonts
   community/index

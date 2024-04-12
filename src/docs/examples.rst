Examples
=============

Math example
************
.. math::

   (a + b)^2  &=  (a + b)(a + b) \\
              &=  a^2 + 2ab + b^2

Code blocks
*******************
  .. code-block:: JSON

    {
      "key": "value"
    }

  .. code-block:: python

    pygments_style = 'sphinx'


  .. code-block:: ruby

    print "Hello, World!\n"

Integrate code to sphinx docs
**************************************************
.. code-block:: python

    """"Example how to integrate code snippets into the docs."""
    import numpy as np
    import pandas as pd
    from .utils import do_something

    while True:
        do_something()
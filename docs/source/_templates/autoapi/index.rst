API Reference
=============

This page contains auto-generated API reference documentation for MONARCHS. [#f1]_.
This is primarily intended for people who are looking to edit or extend the code in some way.
If some documentation is ambiguous or missing, please contact the model maintainers.

This page can be a little overwhelming - use the sidebar buttons to navigate through the specific submodules to find
the information in a slightly more digestible form.

.. toctree::
   {% for page in pages|selectattr("is_top_level_object") %}
   {{ page.include_path }}
   {% endfor %}

.. [#f1] Created with `sphinx-autoapi <https://github.com/readthedocs/sphinx-autoapi>`_

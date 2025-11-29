Contributing
============

Contributions are welcome! Please feel free to submit a Pull Request.

How to Contribute
-----------------

1. Fork the repository
2. Create your feature branch (``git checkout -b feature/AmazingFeature``)
3. Commit your changes (``git commit -m 'Add some AmazingFeature'``)
4. Push to the branch (``git push origin feature/AmazingFeature``)
5. Open a Pull Request

Development Setup
-----------------

To set up a development environment:

.. code-block:: bash

   # Clone the repository
   git clone https://github.com/Nick-Doulos/pyehicle.git
   cd pyehicle

   # Install in editable mode
   pip install -e .

   # Install development dependencies
   pip install pytest pytest-cov black mypy

License
-------

This project is licensed under the MIT License - see the LICENSE file for details.

Support
-------

For questions, bug reports, or feature requests, please open an issue on `GitHub <https://github.com/Nick-Doulos/pyehicle/issues>`_.

Citation
--------

If you use Pyehicle in your research, please cite:

.. code-block:: bibtex

   @software{pyehicle2024,
     title = {Pyehicle: A Python Library for GPS Trajectory Processing and Reconstruction},
     author = {Doulos, Nick},
     year = {2024},
     url = {https://github.com/Nick-Doulos/pyehicle}
   }

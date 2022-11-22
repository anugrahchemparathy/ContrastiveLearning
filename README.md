To run:

```shell
pip install -e . # Installs the internal ldcl package
cd run
python main.py
```

If you make new submodules within the ``ldcl`` directory, you will have to update ``setup.py``. (Likely, not tested, just in case we experience bugs from this in the future.)
* Go to ``setup.py`` and add the submodule in the line ``packages=[...]``
* Redownload by going to root and typing ``pip install -e .``

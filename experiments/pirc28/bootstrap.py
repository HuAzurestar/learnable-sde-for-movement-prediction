"""Internal source-bound entry for the same admitted PIRC-28 worker.

The shared supervisor still owns admission, containment and accounting. This
script changes only module lookup; it grants no standalone execution authority.
"""

from pathlib import Path
import runpy
import sys


def main():
    # Parent sys.path mutations are not inherited by a new interpreter. Bind
    # this child's actual source tree without relying on cwd, an editable
    # installation or PYTHONPATH. Keep the worker's CLI and all its guards.
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    runpy.run_module('experiments.pirc28.worker', run_name='__main__', alter_sys=True)


if __name__ == '__main__':
    main()

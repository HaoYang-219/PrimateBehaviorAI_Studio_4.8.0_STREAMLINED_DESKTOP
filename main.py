from __future__ import annotations
import sys

if __name__ == '__main__':
    if '--self-test' in sys.argv:
        from primate_ai.self_test import run_self_test
        run_self_test()
    else:
        from primate_ai.app import run
        run()

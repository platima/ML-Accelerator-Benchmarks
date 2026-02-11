"""
CLI dispatcher for the utils package.

Running ``python -m utils`` prints usage help.  Running individual
modules (e.g. ``python -m utils.benchmark_analyzer``) triggers their
own ``main()`` functions — but that path causes a RuntimeWarning
because ``__init__.py`` eagerly imports the module before ``runpy``
can execute it.  This ``__main__.py`` provides a clean alternative::

    python -m utils analyse
    python -m utils validate
    python -m utils visualise
    python -m utils detect
"""

import sys


def _print_usage():
    print("Usage: python -m utils <command>")
    print("")
    print("Commands:")
    print("  analyse    Run benchmark_analyzer — compare and report results")
    print("  validate   Run validate_results  — check result files against schema")
    print("  visualise  Run visualization     — generate matplotlib charts")
    print("  detect     Run hardware_detect   — print hardware summary")
    print("")
    print("Example: python -m utils analyse --results-dir results/")


def main():
    args = sys.argv[1:]

    if not args or args[0] in ("-h", "--help"):
        _print_usage()
        sys.exit(0)

    command = args[0]
    # Strip the sub-command so the target module sees a clean argv
    sys.argv = [sys.argv[0] + " " + command] + args[1:]

    if command == "analyse":
        from utils.benchmark_analyzer import main as _main
        _main()
    elif command == "validate":
        from utils.validate_results import main as _main
        _main()
    elif command == "visualise":
        from utils.visualization import main as _main
        _main()
    elif command == "detect":
        from utils.hardware_detect import main as _main
        _main()
    else:
        print("Unknown command: {}".format(command))
        _print_usage()
        sys.exit(1)


if __name__ == "__main__":
    main()

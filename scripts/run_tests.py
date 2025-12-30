import os
import sys
import unittest
from pathlib import Path



def discover_and_run_tests(start_dir: Path) -> int:
    """
    Discover and run all *_test.py files starting from start_dir using unittest.
    """
    start_dir = start_dir.resolve()
    if not start_dir.is_dir():
        print(f"Start directory does not exist or is not a directory: {start_dir}", file=sys.stderr)
        return 1

    # Use unittest discovery with pattern *_test.py
    loader = unittest.TestLoader()
    suite = loader.discover(start_dir=str(start_dir), pattern="*_test.py")

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Return non-zero exit code if any test failed or had errors
    return 0 if result.wasSuccessful() else 1


def main() -> None:
    script_path = Path(__file__).resolve()
    project_root = script_path.parent.parent

    if len(sys.argv) > 1:
        root_dir = Path(sys.argv[1]).resolve()
    else:
        root_dir = project_root

    exit_code = discover_and_run_tests(root_dir)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
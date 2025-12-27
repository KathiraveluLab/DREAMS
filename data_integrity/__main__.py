"""Allow running as: python -m data_integrity"""

from .validator import main
import sys

if __name__ == "__main__":
    sys.exit(main())

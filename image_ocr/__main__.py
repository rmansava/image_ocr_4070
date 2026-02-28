"""Allow running as: python -m image_ocr"""

import sys
from .cli import main

sys.exit(main())

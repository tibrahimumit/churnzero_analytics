"""
ChurnZero-Style Customer Churn Software Solutions Engine
Automated End-to-End Pipeline for Customer Health Scoring, Risk Diagnostics,
What-If Playbook Simulation, and Proactive Retention.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Add project root to sys.path
root_dir = str(Path(__file__).parent.parent)
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from churn_client_lib import *

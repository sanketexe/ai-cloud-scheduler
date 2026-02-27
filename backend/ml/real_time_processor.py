"""
Real-time Feature Extraction Pipeline

Processes streaming cost data for real-time anomaly detection.
Optimized for low-latency feature extraction and data preparation.
"""

import asyncio
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import structlog
from collections import deque
import threading
import queue

from .feature_engine import FeatureEngine, FeatureSet, FeatureConfig
from .data_pipeline import DataPipeline

logger = structlog.get_logger(__name__)


@dataclass
class StreamingDataPoint:
    """Individual streaming data point"""
    timestamp: datetime
    account_id: str
    service: str
    region: str
    cost_amount: float
    usage_amount: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ProcessingWindow:
    """Time window for streaming data processing"""
    window_size_minutes: int = 15
    slide_interval_minutes: int = 5
    max_points_per_window: int = 100
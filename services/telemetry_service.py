"""
Telemetry Tracking and Time-Series Metric Service.
Aggregates prompts, responses, errors, and token volumes across models,
intervals (1 min, 15 min, 1 hr, 1 day), and dynamic time ranges.
"""
import json
import os
import threading
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from config import TELEMETRY_FILE

_telemetry_lock = threading.Lock()

class TelemetryService:
    def __init__(self, file_path=TELEMETRY_FILE):
        self.file_path = file_path
        self._ensure_file()

    def _ensure_file(self):
        with _telemetry_lock:
            if not os.path.exists(self.file_path) or os.path.getsize(self.file_path) == 0:
                with open(self.file_path, "w", encoding="utf-8") as f:
                    json.dump([], f, indent=2)

    def record_event(
        self,
        event_type: str,  # 'prompt', 'response', 'error'
        model: str,
        input_tokens: int = 0,
        output_tokens: int = 0,
        latency_ms: float = 0.0,
        timestamp: Optional[datetime] = None
    ):
        """
        Record a telemetry metric data point.
        """
        now = timestamp or datetime.now()
        point = {
            "timestamp": now.isoformat(),
            "event_type": event_type,
            "model": model,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "latency_ms": latency_ms
        }

        with _telemetry_lock:
            try:
                data = []
                if os.path.exists(self.file_path) and os.path.getsize(self.file_path) > 0:
                    with open(self.file_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                data.append(point)
                with open(self.file_path, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2)
            except Exception as e:
                print(f"[Telemetry Record Error] {e}")

    def get_all_records(self) -> List[Dict[str, Any]]:
        with _telemetry_lock:
            if not os.path.exists(self.file_path) or os.path.getsize(self.file_path) == 0:
                return []
            try:
                with open(self.file_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                return []

    def get_used_models(self) -> List[str]:
        records = self.get_all_records()
        models = set()
        for r in records:
            m = r.get("model")
            if m:
                models.add(m)
        return sorted(list(models))

    def get_aggregated_metrics(
        self,
        model_filter: str = "All Models",
        time_range: str = "1 day",
        interval: str = "15 min",
        custom_start: Optional[str] = None,
        custom_end: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Calculate totals and time-series bucketed series for line plots.
        Intervals: '1 min', '15 min', '1 hr', '1 day'
        Time ranges: 'Last hr', '1 day', 'Week', 'Month', 'Custom'
        """
        records = self.get_all_records()
        now = datetime.now()

        # Determine start and end cutoff dates
        if time_range == "Last hr":
            start_dt = now - timedelta(hours=1)
            end_dt = now
        elif time_range == "Week":
            start_dt = now - timedelta(days=7)
            end_dt = now
        elif time_range == "Month":
            start_dt = now - timedelta(days=30)
            end_dt = now
        elif time_range == "Custom" and custom_start and custom_end:
            try:
                start_dt = datetime.fromisoformat(custom_start)
                end_dt = datetime.fromisoformat(custom_end)
            except Exception:
                start_dt = now - timedelta(days=1)
                end_dt = now
        else:  # default '1 day'
            start_dt = now - timedelta(days=1)
            end_dt = now

        # Determine bucket size in seconds
        interval_seconds = 15 * 60  # default 15 min
        if interval == "1 min":
            interval_seconds = 60
        elif interval == "15 min":
            interval_seconds = 15 * 60
        elif interval == "1 hr":
            interval_seconds = 3600
        elif interval == "1 day":
            interval_seconds = 86400

        # Filter records by model and time
        filtered = []
        for r in records:
            if model_filter != "All Models" and r.get("model") != model_filter:
                continue
            try:
                rt = datetime.fromisoformat(r.get("timestamp"))
                if start_dt <= rt <= end_dt:
                    filtered.append((rt, r))
            except Exception:
                continue

        # Compute overall KPI totals
        total_prompts = sum(1 for _, r in filtered if r.get("event_type") == "prompt")
        total_responses = sum(1 for _, r in filtered if r.get("event_type") == "response")
        total_errors = sum(1 for _, r in filtered if r.get("event_type") == "error")
        total_input_tokens = sum(r.get("input_tokens", 0) for _, r in filtered)
        total_output_tokens = sum(r.get("output_tokens", 0) for _, r in filtered)

        # Generate bucket timestamps from start_dt to end_dt
        bucket_timestamps = []
        cur_ts = int(start_dt.timestamp())
        end_ts = int(end_dt.timestamp())
        # Ensure at least 1 and at most 60 buckets for clean chart rendering
        while cur_ts <= end_ts:
            bucket_timestamps.append(cur_ts)
            cur_ts += interval_seconds

        if not bucket_timestamps:
            bucket_timestamps = [int(start_dt.timestamp()), int(end_dt.timestamp())]

        # Initialize bucket accumulators
        prompts_series = [0] * len(bucket_timestamps)
        responses_series = [0] * len(bucket_timestamps)
        errors_series = [0] * len(bucket_timestamps)
        input_tokens_series = [0] * len(bucket_timestamps)
        output_tokens_series = [0] * len(bucket_timestamps)

        for rt, r in filtered:
            r_ts = int(rt.timestamp())
            # Find closest bucket index
            idx = int((r_ts - int(start_dt.timestamp())) / interval_seconds)
            if 0 <= idx < len(bucket_timestamps):
                ev = r.get("event_type")
                if ev == "prompt":
                    prompts_series[idx] += 1
                elif ev == "response":
                    responses_series[idx] += 1
                elif ev == "error":
                    errors_series[idx] += 1

                input_tokens_series[idx] += r.get("input_tokens", 0)
                output_tokens_series[idx] += r.get("output_tokens", 0)

        # Format labels
        labels = []
        for ts in bucket_timestamps:
            dt_obj = datetime.fromtimestamp(ts)
            if interval == "1 day":
                labels.append(dt_obj.strftime("%b %d"))
            elif interval in ["1 hr", "15 min"]:
                labels.append(dt_obj.strftime("%H:%M"))
            else:
                labels.append(dt_obj.strftime("%H:%M:%S"))

        return {
            "totals": {
                "total_prompts": total_prompts,
                "total_responses": total_responses,
                "total_errors": total_errors,
                "total_input_tokens": total_input_tokens,
                "total_output_tokens": total_output_tokens
            },
            "charts": {
                "labels": labels,
                "prompts": prompts_series,
                "responses": responses_series,
                "errors": errors_series,
                "input_tokens": input_tokens_series,
                "output_tokens": output_tokens_series
            },
            "models_used": self.get_used_models()
        }

# Global singleton
telemetry_service = TelemetryService()

"""
Billing monitor for Google API usage tracking.
Warns when spending thresholds are exceeded.
"""
import logging
import time
from typing import Optional

logger = logging.getLogger(__name__)

# Approximate costs per 1000 units (as of 2026)
COSTS = {
    "gemini_live_audio_input_min": 0.0375,      # $0.0375 per minute
    "gemini_live_audio_output_min": 0.15,       # $0.15 per minute
    "gemini_flash_text_input_1m": 0.075,        # $0.075 per 1M input tokens
    "gemini_flash_text_output_1m": 0.30,        # $0.30 per 1M output tokens
    "gemini_flash_image_input": 0.00026,        # $0.00026 per image
    "gemini_flash_image_output": 0.016,         # $0.016 per image (stylization)
    "veo_video_sec": 0.30,                      # $0.30 per second of video
    "vision_api_label": 0.0015,                 # $1.50 per 1000 images
    "gradium_stt_sec": 0.000003,                # $3 per 1000 seconds (3 credits/sec)
    "gradium_tts_char": 0.000001,               # $1 per 1M characters (1 credit/char)
}

# Warning thresholds in USD
THRESHOLDS = [3.0, 5.0, 9.0, 12.0, 15.0, 19.0, 21.0, 23.0, 24.9]


class BillingMonitor:
    """Track API usage and warn at spending thresholds."""
    
    def __init__(self):
        self.total_cost = 0.0
        self.usage_log = []
        self.warnings_sent = set()
        self.session_start = time.time()
        
    def log_usage(self, api: str, units: float, description: str = ""):
        """Record API usage and calculate cost."""
        cost_key = api
        if cost_key not in COSTS:
            logger.warning(f"Unknown API cost key: {api}")
            return
        
        cost = units * COSTS[cost_key]
        self.total_cost += cost
        
        self.usage_log.append({
            "timestamp": time.time(),
            "api": api,
            "units": units,
            "cost_usd": cost,
            "description": description,
            "running_total": self.total_cost,
        })
        
        # Check thresholds
        self._check_thresholds()
        
    def _check_thresholds(self):
        """Check if any spending threshold has been crossed."""
        for threshold in THRESHOLDS:
            if self.total_cost >= threshold and threshold not in self.warnings_sent:
                self.warnings_sent.add(threshold)
                logger.warning(
                    f"⚠️  BILLING ALERT: Spending has exceeded ${threshold:.2f}! "
                    f"Current total: ${self.total_cost:.4f}"
                )
                # Log recent high-cost items
                recent = [
                    entry for entry in self.usage_log[-20:]
                    if entry["cost_usd"] > 0.01
                ]
                if recent:
                    logger.warning("Recent high-cost operations:")
                    for entry in recent[-5:]:
                        logger.warning(
                            f"  - {entry['api']}: ${entry['cost_usd']:.4f} ({entry['description']})"
                        )
                
    def get_summary(self) -> dict:
        """Return current spending summary."""
        runtime_min = (time.time() - self.session_start) / 60.0
        return {
            "total_cost_usd": self.total_cost,
            "runtime_minutes": runtime_min,
            "warnings_triggered": sorted(list(self.warnings_sent)),
            "next_threshold": self._next_threshold(),
            "operations_count": len(self.usage_log),
        }
    
    def _next_threshold(self) -> Optional[float]:
        """Get the next unwarned threshold."""
        for threshold in THRESHOLDS:
            if threshold not in self.warnings_sent:
                return threshold
        return None
    
    def estimate_gemini_live_call(self, duration_min: float) -> float:
        """Estimate cost of a Gemini Live call."""
        # Assume symmetric input/output
        input_cost = duration_min * COSTS["gemini_live_audio_input_min"]
        output_cost = duration_min * COSTS["gemini_live_audio_output_min"]
        return input_cost + output_cost
    
    def estimate_storybook(self, num_screenshots: int = 5) -> float:
        """Estimate cost of storybook generation."""
        # Input: screenshots + text prompt
        # Output: text + generated images
        input_cost = num_screenshots * COSTS["gemini_flash_image_input"]
        # Assume 1000 input tokens, 2000 output tokens
        text_input = 1000 / 1_000_000 * COSTS["gemini_flash_text_input_1m"]
        text_output = 2000 / 1_000_000 * COSTS["gemini_flash_text_output_1m"]
        # Assume 3 generated images
        image_output = 3 * COSTS["gemini_flash_image_output"]
        return input_cost + text_input + text_output + image_output
    
    def estimate_memory_video(self, num_screenshots: int = 5, video_duration_sec: int = 8) -> float:
        """Estimate cost of memory video pipeline."""
        # Stylization: 5 images
        stylize_cost = num_screenshots * COSTS["gemini_flash_image_output"]
        # Video generation: 8 seconds
        video_cost = video_duration_sec * COSTS["veo_video_sec"]
        return stylize_cost + video_cost


# Global monitor instance
monitor = BillingMonitor()


def log_api_call(api: str, units: float, description: str = ""):
    """Public interface to log API usage."""
    monitor.log_usage(api, units, description)


def get_billing_summary() -> dict:
    """Get current billing summary."""
    return monitor.get_summary()


def estimate_costs() -> dict:
    """Get cost estimates for common operations."""
    return {
        "5_min_call": f"${monitor.estimate_gemini_live_call(5.0):.4f}",
        "storybook_5_screenshots": f"${monitor.estimate_storybook(5):.4f}",
        "memory_video_5_screenshots_8sec": f"${monitor.estimate_memory_video(5, 8):.4f}",
    }

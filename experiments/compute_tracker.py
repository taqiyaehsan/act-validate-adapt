"""
M5 Compute Resource Tracker
=============================
Tracks wall-clock time, memory, LLM usage for NeurIPS reproducibility.
"""

import time
import json
import os
import logging
import platform

logger = logging.getLogger(__name__)


class ComputeTracker:
    """Track compute resources across experiment phases."""

    def __init__(self):
        self.phases = {}
        self.current_phase = None
        self.current_start = None
        self.total_start = time.time()

        # System info
        self.system_info = {
            'platform': platform.platform(),
            'processor': platform.processor(),
            'python_version': platform.python_version(),
            'cpu_count': os.cpu_count(),
        }

        # LLM tracking (populated from GPTClient)
        self.llm_info = {
            'model': None,
            'temperature': None,
            'call_count': 0,
            'total_prompt_tokens': 0,
            'total_completion_tokens': 0,
            'total_tokens': 0,
        }

    def start_phase(self, name):
        """Start timing a phase."""
        self.current_phase = name
        self.current_start = time.time()

    def end_phase(self):
        """End current phase, record timing."""
        if self.current_phase and self.current_start:
            elapsed = time.time() - self.current_start
            self.phases[self.current_phase] = {
                'wall_clock_s': round(elapsed, 1),
                'wall_clock_min': round(elapsed / 60, 2),
            }
            logger.info(f"  [{self.current_phase}] {elapsed:.1f}s "
                        f"({elapsed/60:.1f} min)")
        self.current_phase = None
        self.current_start = None

    def record_llm_usage(self, gpt_client):
        """Pull LLM usage stats from GPTClient instance."""
        if gpt_client is None:
            return
        self.llm_info['model'] = 'gpt-3.5-turbo'
        self.llm_info['call_count'] = gpt_client.call_count
        self.llm_info['total_prompt_tokens'] = gpt_client.total_prompt_tokens
        self.llm_info['total_completion_tokens'] = gpt_client.total_completion_tokens
        self.llm_info['total_tokens'] = gpt_client.total_tokens

    def get_peak_memory_mb(self):
        """Get peak memory usage in MB."""
        try:
            import resource
            # maxrss is in bytes on macOS, KB on Linux
            maxrss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            if platform.system() == 'Darwin':
                return round(maxrss / (1024 * 1024), 1)  # bytes → MB
            else:
                return round(maxrss / 1024, 1)  # KB → MB
        except Exception:
            return None

    def summary(self):
        """Return full compute resource summary."""
        total_elapsed = time.time() - self.total_start
        return {
            'total_wall_clock_s': round(total_elapsed, 1),
            'total_wall_clock_min': round(total_elapsed / 60, 1),
            'peak_memory_mb': self.get_peak_memory_mb(),
            'phases': self.phases,
            'llm': self.llm_info,
            'system': self.system_info,
        }

    def save(self, path):
        """Save to JSON."""
        with open(path, 'w') as f:
            json.dump(self.summary(), f, indent=2)
        logger.info(f"  Compute resources saved to {path}")

    def format_for_summary(self, n_seeds=3):
        """Format for summary.txt appendix."""
        s = self.summary()
        lines = [
            "COMPUTE RESOURCES (M5)",
            "-" * 40,
            f"  Total wall-clock: {s['total_wall_clock_min']:.1f} minutes",
            f"  Peak memory: {s['peak_memory_mb']} MB",
            f"  Seeds: {n_seeds}",
        ]
        if s['phases']:
            lines.append("  Per-phase timing:")
            for phase, info in s['phases'].items():
                if 'wall_clock_min' in info:
                    lines.append(f"    {phase}: {info['wall_clock_min']:.1f} min")
                else:
                    # Enhanced detail entries (from enhance_compute_tracker)
                    detail = ', '.join(f'{k}={v}' for k, v in info.items())
                    lines.append(f"    {phase}: {detail}")
        if s['llm']['model']:
            lines.append(f"  LLM model: {s['llm']['model']}")
            lines.append(f"  LLM calls: {s['llm']['call_count']}")
            lines.append(f"  LLM tokens: {s['llm']['total_tokens']} "
                         f"(prompt={s['llm']['total_prompt_tokens']}, "
                         f"completion={s['llm']['total_completion_tokens']})")
        lines.append(f"  Platform: {s['system']['platform']}")
        lines.append(f"  CPUs: {s['system']['cpu_count']}")
        return '\n'.join(lines)

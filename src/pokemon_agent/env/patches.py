"""Process-wide patches to poke-env and asyncio, applied explicitly instead of on import.

Each command calls apply_all() once at start-up, and so does RolloutWorker.__init__,
because every Ray actor is a separate process. All patches are idempotent.
"""
import functools
import logging
import sys
import traceback

import poke_env
from poke_env.player.baselines import SimpleHeuristicsPlayer

from pokemon_agent.env.heuristics import corrected_stat_estimation
from pokemon_agent.obs.poke_env_compat import apply_request_fixes

logger = logging.getLogger(__name__)


def use_corrected_heuristic():
    """Make SimpleHeuristicsPlayer use the level- and EV-aware stat estimate."""
    SimpleHeuristicsPlayer._stat_estimation = corrected_stat_estimation


def patch_invalid_choice():
    """Answer Showdown's "[Invalid choice]" errors with /choose default, and log handler crashes loudly."""
    original_handle = poke_env.ps_client.PSClient._handle_message
    if getattr(original_handle, "_pokemon_agent_patch", False):
        return

    async def loud_handle(self, message):
        try:
            if "|error|[Invalid choice]" in message:
                room_id = message.split("\n")[0].replace(">", "").strip()
                logger.debug(f"NN chose invalid move in {room_id}. Forcing /choose default.")
                await self.send_message("/choose default", room=room_id)
            await original_handle(self, message)
        except Exception:
            logger.critical(f"Poke-env crash on message: {message[:100]}...")
            traceback.print_exc()
            raise

    loud_handle._pokemon_agent_patch = True
    poke_env.ps_client.PSClient._handle_message = loud_handle


def silence_windows_pipe_errors():
    """On Windows, ignore the proactor transport errors raised when a socket closes mid-write."""
    if sys.platform != "win32":
        return
    import asyncio.proactor_events

    transport = asyncio.proactor_events._ProactorBaseWritePipeTransport
    if getattr(transport._loop_writing, "_pokemon_agent_patch", False):
        return

    def silence_proactor_error(func):
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            try: return func(self, *args, **kwargs)
            except (AssertionError, OSError): pass
        wrapper._pokemon_agent_patch = True
        return wrapper

    transport._loop_writing = silence_proactor_error(transport._loop_writing)


def apply_all():
    use_corrected_heuristic()
    patch_invalid_choice()
    silence_windows_pipe_errors()
    apply_request_fixes()

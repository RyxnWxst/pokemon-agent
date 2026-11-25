"""Showdown connection helpers: server addresses, bot names, login and chat commands."""
import asyncio

from poke_env import ServerConfiguration


def make_server_conf(host: str, port: int) -> ServerConfiguration:
    ws_url = f"ws://{host}:{port}/showdown/websocket"
    http_action = f"http://{host}:{port}/action.php?"
    return ServerConfiguration(ws_url, http_action)


def battle_tag_for(battle) -> str:
    tag = getattr(battle, "battle_tag", None)
    if not tag: tag = f"pyid_{id(battle)}"
    return str(tag)


def mk_name(run_tag: str, p: int, side: str) -> str:
    return f"p{run_tag}{p:03d}{side}"


async def wait_for_login(player, timeout_s: float = 30.0) -> None:
    c = getattr(player, "ps_client", None)
    if c is None: raise RuntimeError(f"{player.username} has no ps_client")
    fn = getattr(c, "wait_for_login", None)
    if not callable(fn): raise RuntimeError(f"{player.username}.ps_client has no wait_for_login()")
    await asyncio.wait_for(fn(), timeout=timeout_s)


async def wait_for_login_retry(player, attempts: int = 5, delay: float = 1.0) -> None:
    last_err = None
    for i in range(attempts):
        try:
            await wait_for_login(player)
            return
        except Exception as e:
            last_err = e
            await asyncio.sleep(delay)
    raise last_err


async def safe_send(player, message: str, room: str, *, retries: int = 8) -> None:
    await wait_for_login(player)
    last_err = None
    for i in range(retries):
        try:
            await player.ps_client.send_message(message, room=room)
            return
        except Exception as e:
            last_err = e
            await asyncio.sleep(0.05 * (i + 1))
    raise RuntimeError(f"[safe_send] failed: {message!r} ({last_err!r})")


async def join_lobby(player) -> None:
    await safe_send(player, "/join lobby", room="")


def rlautospawn_command(player_a: str, player_b: str, battle_format: str, rooms: int) -> str:
    """The rlspawn plugin command that keeps `rooms` battles running between two bots."""
    return f"/rlautospawn {player_a}, {player_b}, {battle_format}, {rooms}"

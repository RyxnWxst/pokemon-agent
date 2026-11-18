"""Compatibility fixes for the installed poke-env request parser."""
from functools import wraps

from poke_env.battle import Pokemon, PokemonType


def apply_request_fixes():
    """Retain the owner's server-provided Tera type without revealing opponents."""
    original = Pokemon.update_from_request
    if getattr(original, "_ps_ppo_tera_fix", False):
        return

    @wraps(original)
    def update_from_request(self, request_pokemon):
        result = original(self, request_pokemon)
        # poke-env 0.15 ignores this side-request field. Assigning the known
        # type must not mark the Pokemon as already Terastallized.
        tera_type = request_pokemon.get("teraType")
        if tera_type:
            self._terastallized_type = PokemonType.from_name(tera_type)
        return result

    update_from_request._ps_ppo_tera_fix = True
    Pokemon.update_from_request = update_from_request

"""Players used outside training: a saved policy (evaluation) and the recording teacher (collection)."""
import numpy as np
import torch
from poke_env.player import DefaultBattleOrder, Player, SimpleHeuristicsPlayer

from pokemon_agent.env.events import ProtocolEventRecorder
from pokemon_agent.obs.actions import map_index_to_order, map_order_to_index
from pokemon_agent.obs.assembler import ObservationAssembler
from pokemon_agent.rl.distributions import masked_sample


class SavedModelPlayer(ProtocolEventRecorder, Player):
    """Plays a fixed checkpoint in-process, one decision at a time (no Ray)."""
    def __init__(self, model, device, greedy=False, progress=None, **kwargs):
        self.model, self.device, self.greedy = model, device, greedy
        self.progress = progress
        self.assembler = ObservationAssembler()
        self.decisions = 0
        super().__init__(**kwargs)

    async def choose_move(self, battle):
        observation = self.assembler.assemble(battle, events=self.pop_events(battle.battle_tag))
        obs = torch.from_numpy(observation[None]).to(self.device)
        with torch.no_grad():
            logits = self.model(obs)[0]
            start, end = self.assembler.offsets["action_mask"]
            action, *_ = masked_sample(logits, obs[:, start:end], greedy=self.greedy)
        self.decisions += 1
        choice, kwargs = map_index_to_order(int(action), battle)
        return DefaultBattleOrder() if choice == "DEFAULT" else self.create_order(choice, **kwargs)

    def _battle_finished_callback(self, battle):
        self.drop_events(battle.battle_tag)
        super()._battle_finished_callback(battle)
        if self.progress:
            self.progress(self)


class RecordingTeacher(ProtocolEventRecorder, SimpleHeuristicsPlayer):
    """The heuristic, recording (observation, action index) for every decision it makes."""
    def __init__(self, **kwargs):
        self.pending, self.episodes = {}, {}
        self.assembler = ObservationAssembler()
        super().__init__(**kwargs)

    def choose_move(self, battle):
        order = super().choose_move(battle)
        observation = self.assembler.assemble(battle, events=self.pop_events(battle.battle_tag))
        action = map_order_to_index(order, battle)
        start, end = self.assembler.offsets["action_mask"]
        if observation[start:end][action] < .5:
            raise RuntimeError("Teacher action is outside the legal mask")
        choice, kwargs = map_index_to_order(action, battle)
        if choice == "DEFAULT" or self.create_order(choice, **kwargs).message != order.message:
            raise RuntimeError("Teacher action failed its order roundtrip")
        self.pending.setdefault(battle.battle_tag, []).append((observation, action))
        return order

    def _battle_finished_callback(self, battle):
        samples = self.pending.pop(battle.battle_tag)
        observations = torch.from_numpy(np.stack([item[0] for item in samples]))
        actions = torch.tensor([item[1] for item in samples], dtype=torch.long)
        self.episodes[battle.battle_tag] = observations, actions, torch.zeros(len(actions))
        self.drop_events(battle.battle_tag)
        super()._battle_finished_callback(battle)

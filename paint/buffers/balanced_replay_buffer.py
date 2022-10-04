import numpy as np
import math

from .simple_replay_buffer import SimpleReplayBuffer


class BalancedReplayBuffer:
    def __init__(
        self,
        data_specs,
        max_size,
        demo_fraction_generator,
        batch_size=None,
        replay_dir=None,
        discount=0.99,
        filter_transitions=True,
        with_replacement=True,
    ):

        buffer_kwargs = {
            "data_specs": data_specs,
            "max_size": max_size,
            "batch_size": batch_size,
            "replay_dir": replay_dir,
            "discount": discount,
            "filter_transitions": filter_transitions,
            "with_replacement": with_replacement,
        }

        self.demo_buffer = SimpleReplayBuffer(**buffer_kwargs)
        self.regular_buffer = SimpleReplayBuffer(**buffer_kwargs)
        self.buffer_kwargs = buffer_kwargs
        self.demo_fraction_generator = demo_fraction_generator

    def __len__(self):
        return len(self.demo_buffer) + len(self.regular_buffer)

    @property
    def _num_transitions(self):
        """
        Only count transitions regular buffer
        because the replace function takes indices
        in terms of the regular buffer
        """
        return self.regular_buffer._num_transitions

    def replace(self, idx, time_step):
        self.regular_buffer.replace(idx, time_step)

    @property
    def stuck_ratio(self):
        return self.regular_buffer.stuck_ratio

    def add_offline_data(self, demos, default_action):
        self.demo_buffer.add_offline_data(demos=demos, default_action=default_action)

    def add(self, time_step):
        self.regular_buffer.add(time_step=time_step)

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def next(self):
        batch_size = self.buffer_kwargs["batch_size"]
        demo_fraction = self.demo_fraction_generator(len(self.regular_buffer))

        demo_batch_size = math.ceil(batch_size * demo_fraction)
        assert demo_batch_size > 0 and demo_batch_size < batch_size
        regular_batch_size = batch_size - demo_batch_size

        demo_batch = self.demo_buffer.next(
            batch_size=demo_batch_size,
        )
        regular_batch = self.regular_buffer.next(batch_size=regular_batch_size)

        batch = ()
        for demo_element, regular_element in zip(demo_batch, regular_batch):
            new_elem = np.concatenate([demo_element, regular_element], axis=0)
            batch += (new_elem,)

        return batch

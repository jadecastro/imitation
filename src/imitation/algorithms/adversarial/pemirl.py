"""Probabilsitic Embedding Meta Inverse Reinforcement Learning (PEMIRL)."""

import numpy as np
import gym
from typing import Iterable, Optional, Tuple

import torch as th
from torch.distributions import Normal
from torch.distributions.independent import Independent
from stable_baselines3.common import base_class, preprocessing, vec_env

from imitation.algorithms import base
from imitation.algorithms.adversarial import common
from imitation.rewards import reward_nets
from imitation.util import networks


class ContextEncoderNet(th.nn.Module):
    """MLP that takes as input the state, action, next state and done flag.

    These inputs are flattened and then concatenated to one another. Each input
    can enabled or disabled by the `use_*` constructor keyword arguments.
    """

    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        traj_length: int,
        batch_size: int,
        use_state: bool = True,
        use_action: bool = True,
        use_next_state: bool = False,
        use_done: bool = False,
        **kwargs,
    ):
        """Builds an MLP for the context encoder.

        Args:
            observation_space: The observation space.
            action_space: The action space.
            use_state: should the current state be included as an input to the MLP?
            use_action: should the current action be included as an input to the MLP?
            use_next_state: should the next state be included as an input to the MLP?
            use_done: should the "done" flag be included as an input to the MLP?
            kwargs: passed straight through to `build_mlp`.
        """
        super().__init__()
        self.traj_length = traj_length if traj_length is not None else 1
        self.observation_space = observation_space
        self.action_space = action_space
        self.normalize_images = False
        combined_size = 0

        self.use_state = use_state
        if self.use_state:
            combined_size += preprocessing.get_flattened_obs_dim(observation_space)

        self.use_action = use_action
        if self.use_action:
            combined_size += preprocessing.get_flattened_obs_dim(action_space)
        # import IPython; IPython.embed()

        self.use_next_state = use_next_state
        if self.use_next_state:
            # combined_size += preprocessing.get_flattened_obs_dim(observation_space)
            raise ValueError("use_next_state not yet implemented for LSTM.")

        self.use_done = use_done
        if self.use_done:
            # combined_size += 1
            raise ValueError("use_done not yet implemented for LSTM.")

        # input_size = (batch_size, traj_length, combined_size)

        self.num_layers = 1
        self.hidden_size = 32
        self.bidirectional = False
        self.hidden_factor = (2 if self.bidirectional else 1) * self.num_layers

        # import IPython; IPython.embed()

        # Build MLPs for mean and log-std.
        # Note that the MetaIRL repo uses a GaussianMLPPolicy from rllab's sandbox.rocky.tf.policies.
        # Their input is expert_traj_var, of dimension [meta_batch_size, batch_size, T, dO+dU]
        # They also provide a placeholder to make it a recurrent net.
        self.encoder, self.hidden_to_mean, self.hidden_to_logstd = networks.build_gaussian_lstm(
            in_size=combined_size,
            hidden_size=self.hidden_size,
            latent_size=1,
            num_layers=self.num_layers,
            bidirectional=self.bidirectional,
            flatten_input=False,
        )

    def preprocess(
        self,
        state_hist: np.ndarray,
        action_hist: np.ndarray,
        next_state: np.ndarray = None,
        done: np.ndarray = None,
    ) -> Tuple[th.Tensor, th.Tensor, th.Tensor, th.Tensor]:
        """Preprocess a batch of input transitions and convert it to PyTorch tensors.

        The output of this function is suitable for its forward pass,
        so a typical usage would be ``model(*model.preprocess(transitions))``.

        Args:
            state_hist: The observation input history. Its shape is
                `(batch_size, traj_length) + observation_space.shape`.
            action_hist: The action input history. Its shape is
                `(batch_size, traj_length) + action_space.shape`. The None dimension is
                expected to be the same as None dimension from `obs_input`.
            next_state: The observation input. Its shape is
                `(batch_size,) + observation_space.shape`.
            done: Whether the episode has terminated. Its shape is `(batch_size,)`.

        Returns:
            Preprocessed transitions: a Tuple of tensors containing
            observations, actions, next observations and dones.
        """
        state_hist_th = th.as_tensor(state_hist, device=self.device)
        action_hist_th = th.as_tensor(action_hist, device=self.device)
        next_state_th = None
        if next_state is not None:
            next_state_th = th.as_tensor(next_state, device=self.device)
        done_th = None
        if done is not None:
            done_th = th.as_tensor(done, device=self.device)

        del state_hist, action_hist, next_state, done  # unused

        # preprocess
        state_hist_th = preprocessing.preprocess_obs(
            state_hist_th,
            self.observation_space,
            self.normalize_images,
        )
        action_hist_th = preprocessing.preprocess_obs(
            action_hist_th,
            self.action_space,
            self.normalize_images,
        )
        if next_state_th is not None:
            next_state_th = preprocessing.preprocess_obs(
                next_state_th,
                self.observation_space,
                self.normalize_images,
            )
        if done_th is not None:
            done_th = done_th.to(th.float32)

        n_gen = len(state_hist_th)
        if next_state_th is not None:
            assert state_hist_th.shape[2:] == next_state_th.shape[1:]
        assert len(action_hist_th) == n_gen

        return state_hist_th, action_hist_th, next_state_th, done_th

    def forward(
            self,
            state_hist: th.Tensor,
            action_hist: th.Tensor,
            next_state: th.Tensor,
            done: th.Tensor
    ) -> Tuple[th.Tensor, th.Tensor]:
        inputs = []
        if self.use_state:
            inputs.append(th.flatten(state_hist, 2))
        if self.use_action:  # N.B. Default highway-env uses a discrete (one-hot) encoding.
            inputs.append(th.flatten(action_hist, 2))
        # if self.use_next_state:
        #     inputs.append(th.flatten(next_state, 1))
        # if self.use_done:
        #     inputs.append(th.reshape(done, [-1, 1]))

        inputs_concat = th.cat(inputs, dim=2)
        # import IPython; IPython.embed()

        _, (hidden, state) = self.encoder(inputs_concat)
        if self.bidirectional or self.num_layers > 1:
            # Flatten the hidden layer.
            hidden = hidden.view(-1, self.hidden_size * self.hidden_factor)
        else:
            hidden = hidden.squeeze()
        mean_z = self.hidden_to_mean(hidden)
        log_std_z = self.hidden_to_logstd(hidden)
        return mean_z, log_std_z

    @property
    def device(self) -> th.device:
        """Heuristic to determine which device this module is on."""
        try:
            first_param = next(self.parameters())
            return first_param.device
        except StopIteration:
            # if the model has no parameters, we use the CPU
            return th.device("cpu")


class PEMIRL(common.AdversarialTrainer):
    """Probabilistic Embedding Meta Inverse Reinforcement Learning (`PEMIRL`_).

    .. _PEMIRL: https://arxiv.org/pdf/1909.09314.pdf
    """

    def __init__(
        self,
        *,
        demonstrations: base.AnyTransitions,
        demo_batch_size: int,
        venv: vec_env.VecEnv,
        gen_algo: base_class.BaseAlgorithm,
        reward_net: Optional[reward_nets.RewardNet] = None,
        context_encoder_net: Optional[th.nn.Module] = None,
        traj_length: int = None,
        **kwargs,
    ):
        """Builds an PEMIRL trainer.

        Args:
            demonstrations: Demonstrations from an expert (optional). Transitions
                expressed directly as a `types.TransitionsMinimal` object, a sequence
                of trajectories, or an iterable of transition batches (mappings from
                keywords to arrays containing observations, etc).
            demo_batch_size: The number of samples in each batch of expert data. The
                discriminator batch size is twice this number because each discriminator
                batch contains a generator sample for every expert sample.
            venv: The vectorized environment to train in.
            gen_algo: The generator RL algorithm that is trained to maximize
                discriminator confusion. Environment and logger will be set to
                `venv` and `custom_logger`.
            reward_net: Reward network; used as part of AIRL discriminator. Defaults to
                `reward_nets.BasicShapedRewardNet` when unspecified.
            traj_length: Trajectory length to use for training PEMIRL LSTM encoder.
            **kwargs: Passed through to `AdversarialTrainer.__init__`.

        Raises:
            TypeError: If `gen_algo.policy` does not have an `evaluate_actions`
                attribute (present in `ActorCriticPolicy`), needed to compute
                log-probability of actions.
        """

        # TODO(jon): Make sure these work for a discrete action space (highway-env's default).
        if reward_net is None:
            reward_net = reward_nets.BasicShapedRewardNet(
                observation_space=venv.observation_space,
                action_space=venv.action_space,
            )
        self._reward_net = reward_net
        if context_encoder_net is None:
            context_encoder_net = ContextEncoderNet(
                observation_space=venv.observation_space,
                action_space=venv.action_space,
                traj_length=traj_length,
                batch_size=demo_batch_size,
            )
        self._context_encoder_net = context_encoder_net
        print(self._reward_net.parameters())
        print(self._context_encoder_net.parameters())
        print(f"traj_length: {traj_length}")
        super().__init__(
            demonstrations=demonstrations,
            demo_batch_size=demo_batch_size,
            venv=venv,
            gen_algo=gen_algo,
            disc_parameters=list(self._reward_net.parameters()) + list(self._context_encoder_net.parameters()),
            traj_length=traj_length,
            **kwargs,
        )
        if not hasattr(self.gen_algo.policy, "evaluate_actions"):
            raise TypeError(
                "AIRL needs a stochastic policy to compute the discriminator output.",
            )

    # *** TODO *** Add context variable 
    def logits_gen_is_high(
        self,
        state: th.Tensor,
        action: th.Tensor,
        next_state: th.Tensor,
        done: th.Tensor,
        log_policy_act_prob: th.Tensor,
        reparam_latent: th.Tensor,
    ) -> Tuple[th.Tensor, th.Tensor]:
        """Compute the discriminator's logits for each state-action sample."""
        if log_policy_act_prob is None:
            raise TypeError(
                "Non-None `log_policy_act_prob` is required for this method.",
            )
        reward_output_train = self._reward_net(state, action, next_state, done)
        # In Fu's AIRL paper (https://arxiv.org/pdf/1710.11248.pdf), the
        # discriminator output was given as exp(r_theta(s,a)) /
        # (exp(r_theta(s,a)) + log pi(a|s)), with a high value corresponding to
        # expert and a low value corresponding to generator (the opposite of
        # our convention).
        #
        # Observe that sigmoid(log pi(a|s) - r(s,a)) = exp(log pi(a|s) -
        # r(s,a)) / (1 + exp(log pi(a|s) - r(s,a))). If we multiply through by
        # exp(r(s,a)), we get pi(a|s) / (pi(a|s) + exp(r(s,a))). This is the
        # original AIRL discriminator expression with reversed logits to match
        # our convention of low = expert and high = generator (like GAIL).

        return log_policy_act_prob - reward_output_train, reward_output_train

    def reparameterize(self, mean, log_std):
        eps = th.randn_like(mean)
        return mean + th.exp(log_std) * eps

    def kld(self, mean, log_var):
        # TODO(jon): This formulation uses log_var - we need to reformulate for log_std.
        return -0.5 * th.sum(1 + log_var - mean.pow(2) - log_var.exp())

    def context_encoder_log_likelihood(
        self,
        state_hist: th.Tensor,
        action_hist: th.Tensor,
        next_state: th.Tensor,
        done: th.Tensor,
        context_id,
    ) -> Tuple[th.Tensor, th.Tensor]:
        # Note: We're inputting the expert trajectory.
        mean_context, log_std_context = self._context_encoder_net(
            state_hist, action_hist, next_state, done
        )

        # Manually dump the context var info to stdout for quick insepection.
        log_data = dict(
            mean_context=mean_context,
            log_std_context=log_std_context,
            context_id=context_id,
        )

        # Quickly print some of the labels to inspect if we're on the right track.
        max_count = 100
        count = 0
        for i, cid in enumerate(context_id):
            if not np.isnan(cid):
                print(f"gen context {mean_context[i]}, GT context {cid}")
                count += 1
            if count > max_count:
                break

        # Reparameterization step
        # kld = self.kld(mean_context, log_std_context)
        reparam_latent = self.reparameterize(mean_context, log_std_context)  # [num_episodes, latent_dim]
        std_context = th.exp(log_std_context)
        normal_dist = Normal(mean_context, std_context)
        # Makes it so that a sample from the distribution is treated as a
        # single sample and not dist.batch_shape samples.
        normal_dist = Independent(normal_dist, 1)
        log_q_m_tau = normal_dist.log_prob(reparam_latent)

        return log_q_m_tau, reparam_latent, log_data

    @property
    def reward_train(self) -> reward_nets.RewardNet:
        return self._reward_net

    @property
    def reward_test(self) -> reward_nets.RewardNet:
        if isinstance(self._reward_net, reward_nets.ShapedRewardNet):
            return self._reward_net.base
        else:
            return self._reward_net

    @property
    def context_encoder_train(self) -> th.nn.Module:
        return self._context_encoder_net

    @property
    def context_encoder_test(self) -> th.nn.Module:
        return self._context_encoder_net

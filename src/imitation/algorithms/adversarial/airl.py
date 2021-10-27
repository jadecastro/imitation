"""Adversarial Inverse Reinforcement Learning (AIRL)."""

from typing import Optional

import torch as th
from torch.distributions import Normal
from torch.distributions.independent import Independent
from stable_baselines3.common import base_class, vec_env

from imitation.algorithms import base
from imitation.algorithms.adversarial import common
from imitation.rewards import reward_nets
from imitation.util import networks


class ContextEncoderNet(th.nn.Module):
    """Simple implementation of a potential using an MLP."""

    def __init__(self, observation_space: gym.Space, hid_sizes: Iterable[int]):
        """Initialize the potential.

        Args:
            observation_space: observation space of the environment.
            hid_sizes: widths of the hidden layers of the MLP.
        """
        super().__init__()
        input_size = preprocessing.get_flattened_obs_dim(observation_space)
        self._mean_net = networks.build_mlp(
            in_size=input_size,
            hid_sizes=hid_sizes,
            squeeze_output=True,
            flatten_input=True,
        )
        self._std_net = networks.build_mlp(
            in_size=input_size,
            hid_sizes=hid_sizes,
            squeeze_output=True,
            flatten_input=True,
        )

    def forward(self, state: th.Tensor) -> th.Tensor:
        mean = self._mean_net(state)
        log_std = self._std_net(state)
        return mean, log_std


class AIRL(common.AdversarialTrainer):
    """Adversarial Inverse Reinforcement Learning (`AIRL`_).

    .. _AIRL: https://arxiv.org/abs/1710.11248
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
        **kwargs,
    ):
        """Builds an AIRL trainer.

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
            **kwargs: Passed through to `AdversarialTrainer.__init__`.

        Raises:
            TypeError: If `gen_algo.policy` does not have an `evaluate_actions`
                attribute (present in `ActorCriticPolicy`), needed to compute
                log-probability of actions.
        """
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
            )
        self._context_encoder_net = context_encoder_net
        super().__init__(
            demonstrations=demonstrations,
            demo_batch_size=demo_batch_size,
            venv=venv,
            gen_algo=gen_algo,
            disc_parameters=self._reward_net.parameters() + self._enc_net.parameters(),
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
    ) -> th.Tensor:
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

    def reparameterize(self, mu, log_std):
        eps = torch.randn_like(mean)
        return mean + torch.exp(0.5 * log_std) * eps
    
    def kld(self, mean, log_std):
        return -0.5 * torch.sum(1 + log_std - mean.pow(2) - log_std.exp())
    
    def context_encoder_log_likelihood(
        self,
        state: th.Tensor,
        action: th.Tensor,
        next_state: th.Tensor,
        done: th.Tensor,
    ) -> th.Tensor:
        # Note: We're inputting the expert trajectory.
        mean_context, log_std_context = self._context_encoder_net(state, action, next_state, done)

        # Reparameterization step
        # kld = self.kld(mean_context, log_std_context)
        reparam_latent = self.reparameterize(mean_context, log_std_context)  # [num_episodes, latent_dim]
        normal_dist = Normal(mean_context, log_std_context)
        # Makes it so that a sample from the distribution is treated as a
        # single sample and not dist.batch_shape samples.
        dist = Independent(dist, 1)
        log_q_m_tau = normal_dist.log_prob(reparam_latent)

        return log_q_m_tau

    @property
    def reward_train(self) -> reward_nets.RewardNet:
        return self._reward_net

    @property
    def reward_test(self) -> reward_nets.RewardNet:
        if isinstance(self._reward_net, reward_nets.ShapedRewardNet):
            return self._reward_net.base
        else:
            return self._reward_net

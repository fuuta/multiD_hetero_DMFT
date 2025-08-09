from functools import partial
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax import random as jr

from ..dataclass.result import TrialResult
from ..dataclass.external_input import OUSourceParams
from ..dataclass.net_params import HeterogeneousParameter, NetworkParameters

def f_DvarNN(
    ys: jnp.ndarray,
    t: jax.Array,
    eI: jnp.ndarray,
    J: jnp.ndarray,
    phi: Callable[[jnp.ndarray], jnp.ndarray],
    coeffs: jnp.ndarray,
):
    phi_ys = phi(ys)
    dys = coeffs @ ys + J @ phi_ys + eI
    return dys

# Helper for RK step, to be used in scan
def _rk_step_scan_body(
    carry_ys: jnp.ndarray, # previous state (ys or ys_dys)
    i_scan_step: jax.Array, # current step index, from 0 to n_timestep_inblock - 2
    dt: float,
    f_dys: Callable[[jnp.ndarray, jax.Array, jnp.ndarray], jnp.ndarray],
    f_eI: Callable[[jax.Array], jnp.ndarray], 
    N: int,
    D: int,
    is_with_dys: bool,
):
    if is_with_dys:
        previous_ys_actual = carry_ys[: N * D]
    else:
        previous_ys_actual = carry_ys

    # t_eval corresponds to `dt * i_timestep` in the original rk_step's logic,
    # where i_timestep was the 1-based index of the state being computed.
    # i_scan_step is 0-based. If i_scan_step = 0, original i_timestep = 1.
    # This t_eval is used as the time argument for f_dys and f_eI.
    # Note: The 't' argument of f_DvarNN (and thus f_dys) is currently unused in its definition.
    t_eval = dt * (i_scan_step + 1)

    k1 = f_dys(previous_ys_actual, t_eval, f_eI(t_eval))
    k2 = f_dys(previous_ys_actual + dt * k1 / 2.0, t_eval + dt / 2.0, f_eI(t_eval + dt / 2.0))
    k3 = f_dys(previous_ys_actual + dt * k2 / 2.0, t_eval + dt / 2.0, f_eI(t_eval + dt / 2.0))
    k4 = f_dys(previous_ys_actual + dt * k3, t_eval + dt, f_eI(t_eval + dt))

    dydt = 1.0 / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    ysi_next = previous_ys_actual + dydt * dt

    if is_with_dys:
        next_state_combined = jnp.hstack([ysi_next, dydt])
    else:
        next_state_combined = ysi_next
    
    return next_state_combined, next_state_combined # (next_carry, scan_output_for_this_step)

# Helper for FE step, to be used in scan
def _fe_step_scan_body(
    carry_ys: jnp.ndarray, # previous state (ys or ys_dys)
    scan_input_tuple: tuple[jnp.ndarray, jax.Array], # (eI_slice_for_current_step, i_scan_step)
    dt: float,
    f_dys: Callable[[jnp.ndarray, jax.Array, jnp.ndarray], jnp.ndarray],
    N: int,
    D: int,
    is_with_dys: bool,
):
    eI_current_step, i_scan_step = scan_input_tuple # i_scan_step from 0 to n_timestep_inblock - 2

    if is_with_dys:
        previous_ys_actual = carry_ys[: N * D]
    else:
        previous_ys_actual = carry_ys

    # t_eval_for_f_dys corresponds to `dt * i_timestep` in the original fe_step's logic.
    # Note: The 't' argument of f_DvarNN (and thus f_dys) is currently unused in its definition.
    t_eval_for_f_dys = dt * (i_scan_step + 1)
    
    # eI_current_step is eI[i_scan_step, :], which is input at time dt * i_scan_step.
    # The original f_dys call was f_dys(previous_ys, t_eval_for_f_dys, eI_current_step).
    dydt = f_dys(previous_ys_actual, t_eval_for_f_dys, eI_current_step)
    ysi_next = previous_ys_actual + dydt * dt

    if is_with_dys:
        next_state_combined = jnp.hstack([ysi_next, dydt])
    else:
        next_state_combined = ysi_next
        
    return next_state_combined, next_state_combined # (next_carry, scan_output_for_this_step)

def _fe_wnoise_step_scan_body(
    carry_ys: jnp.ndarray, # previous state (ys or ys_dys)
    scan_input_tuple: tuple[jnp.ndarray, jnp.ndarray, jax.Array], # (eI_slice_for_current_step, noise, i_scan_step)
    dt: float,
    f_dys: Callable[[jnp.ndarray, jax.Array, jnp.ndarray], jnp.ndarray],
    N: int,
    D: int,
    is_with_dys: bool,
):
    eI_current_step, noise_step, i_scan_step = scan_input_tuple # i_scan_step from 0 to n_timestep_inblock - 2

    if is_with_dys:
        previous_ys_actual = carry_ys[: N * D]
    else:
        previous_ys_actual = carry_ys

    dydt = f_dys(previous_ys_actual, dt * (i_scan_step + 1), eI_current_step)
    ysi_next = previous_ys_actual + dydt * dt + noise_step*jnp.sqrt(2*dt) # noise is added to the next state directly (Euler-Maruyama method)

    if is_with_dys:
        next_state_combined = jnp.hstack([ysi_next, dydt])
    else:
        next_state_combined = ysi_next
        
    return next_state_combined, next_state_combined # (next_carry, scan_output_for_this_step) ノイズは返さない

def initialize_states(
    rndkey: jax.Array, N: int, Ndimstate: int, scale: float = 1e-1
) -> tuple[jnp.ndarray, jax.Array]:
    # 1番目の変数xのみ標準正規分布からi.i.dに初期化, 残りの変数は0で初期化
    initial_x = jr.normal(rndkey, shape=[N]) * scale  # 初期値は小さな値にする
    rndkey, subkey = jax.random.split(rndkey)
    initial_state = jnp.hstack([initial_x, jnp.zeros([Ndimstate - N])])
    return initial_state, rndkey


def expands_coeff_matrix(
    rndkey: jax.Array, coeff: jnp.ndarray, hetero_info: HeterogeneousParameter | None, N: int
) -> tuple[jnp.ndarray, jnp.ndarray | None, jax.Array]:
    nd_coeff = jnp.kron(coeff, jnp.eye(N))  # coeffを N*n_dim x N*n_dim 次元に拡張する
    if hetero_info is not None:
        hetero_vars, rndkey = hetero_info.dist.sample(rndkey, shape=[N])
        hetero_vars += hetero_info.shift  # ヘテロ変数にシフトを加える
        hetero_vars = hetero_vars[jnp.argsort(hetero_vars)] # NOTE: ヘテロ変数は昇順にする
        if hetero_info.index == "gain":
            return nd_coeff, hetero_vars, rndkey
        hetero_matrix = (
            jnp.diag(hetero_vars) * coeff[hetero_info.index[0], hetero_info.index[1]]
        )  # coeff倍されていることに注意 (基本的に1 or -1)
        nd_coeff = nd_coeff.at[
            hetero_info.index[0] * N : (hetero_info.index[0] + 1) * N,
            hetero_info.index[1] * N : (hetero_info.index[1] + 1) * N,
        ].set(hetero_matrix)
        return nd_coeff, hetero_vars, rndkey
    else:
        return nd_coeff, None, rndkey


def run_1trial_ndhetero(
    rndkey: jax.Array,
    n_timestep: int,
    n_block: int,
    dt: float,
    N: int,
    D: int,
    g: float,
    phi: Callable,
    hetero_info: HeterogeneousParameter | None,
    net_params: NetworkParameters,
    initstate_scale: float = 1e-1,
    isret_dy: bool = False,
) -> tuple[TrialResult, jax.Array]:
    assert n_timestep % n_block == 0
    initial_rndkey = rndkey.copy()

    nd_coeff, hetero_var_vals, rndkey = expands_coeff_matrix(
        rndkey, jnp.asarray(net_params.coefficient), hetero_info, N
    )
    J = (
        jr.normal(rndkey, shape=[N, N]) / jnp.sqrt(N) * g
    )
    J = jnp.multiply(
        J, (jnp.ones([N, N]) - jnp.eye(N))
    )
    if hetero_info is not None and hetero_info.index == "gain":
        assert hetero_var_vals is not None, "Heterogeneous variable values must be provided for gain index."
        J = J @ jnp.diag(hetero_var_vals)  # ヘテロ変数を掛ける
    rndkey, subkey = jax.random.split(rndkey)
    nd_J = jnp.zeros(shape=[N * D, N * D])
    nd_J = nd_J.at[:N, :N].set(J)

    partial_f_NN = partial(f_DvarNN, J=nd_J, phi=phi, coeffs=nd_coeff)

    list_ys = []
    list_dys = []
    list_eIs = []
    list_sources = []
    n_timestep_inblock = (
        int(n_timestep / n_block) + 1
    )

    Ndimstate = (1 + (1 if isret_dy else 0)) * N * D

    external_input_dc = net_params.external_input
    if external_input_dc is not None:
        Win, rndkey = external_input_dc.Win_params.create(rndkey, N)
        def eI_generator_wrapper(rndkey: jax.Array, start_ts:int, end_ts:int) -> tuple[jax.Array, jnp.ndarray, jnp.ndarray]: # type: ignore
            return external_input_dc.generate(
                    rndkey=rndkey,
                    Win=Win,
                    start_ts=start_ts,
                    end_ts=end_ts,
                    N=N,
                    D=D,
                )
        if isinstance(external_input_dc.soureparam, OUSourceParams):
            rndkey, _, source = eI_generator_wrapper(rndkey, -100, 0)
            external_input_dc.soureparam.set_init_source(source[-1, :])
    else:
        def eI_generator_wrapper(rndkey: jax.Array, start_ts:int, end_ts:int) -> tuple[jax.Array, jnp.ndarray, jnp.ndarray]:
            # Ensure returned arrays have correct dtype for vstack later if needed, though not strictly necessary if not used.
            return rndkey, jnp.array([], dtype=jnp.float32), jnp.array([], dtype=jnp.float32)
        Win = None

    init_states_val = None # Stores the state from the end of one block to the start of the next
    for i_block in range(n_block):
        if i_block == 0:
            last = -2 
        elif i_block == n_block - 1:
            last = None
        else:
            last = -1

        rndkey, eI_block, sources_block = eI_generator_wrapper(
            rndkey=rndkey,
            start_ts=i_block * n_timestep_inblock,
            end_ts=(i_block + 1) * n_timestep_inblock,
        )

        if init_states_val is None:
            # Initialize the first block with a new state
            current_block_init_carry, rndkey = initialize_states(
                rndkey=rndkey,
                N=N,
                Ndimstate=Ndimstate,
                scale=initstate_scale,  # Use the scale from the parameters
            )
        else:
            # Use the last state from the previous block as the initial state for this block
            current_block_init_carry = init_states_val

        # Number of steps to simulate in this block using scan
        num_scan_steps = n_timestep_inblock - 1


        if external_input_dc is not None or net_params.noise is not None:
            # Prepare inputs for scan: external inputs and corresponding time indices
            # eI_block has shape (n_timestep_inblock, N*D)
            # We need eI for steps 0 to num_scan_steps-1
            xs_eI_slices = eI_block[:-1, :] 
            xs_indices = jnp.arange(num_scan_steps)
            if net_params.noise is not None:
                # If noise is present, we need to add it to the eI input
                xs_noise = jnp.zeros(shape=(num_scan_steps, N * D))
                xs_noise_val, rndkey = net_params.noise.sample(rndkey, shape=(num_scan_steps, N))
                xs_noise = xs_noise.at[:, :N].set(xs_noise_val)  # Only add noise to the first N dimensions
                scan_xs = (xs_eI_slices, xs_noise, xs_indices)

                partial_scan_body = partial(
                    _fe_wnoise_step_scan_body,
                    dt=dt,
                    f_dys=partial_f_NN,
                    N=N,
                    D=D,
                    is_with_dys=isret_dy,
                )
                _, scanned_states_part = lax.scan(
                    partial_scan_body,
                    current_block_init_carry,
                    scan_xs,
                )

            else:
                scan_xs = (xs_eI_slices, xs_indices)

                partial_scan_body = partial(
                    _fe_step_scan_body,
                    dt=dt,
                    f_dys=partial_f_NN,
                    N=N,
                    D=D,
                    is_with_dys=isret_dy,
                )
                _, scanned_states_part = lax.scan(
                    partial_scan_body,
                    current_block_init_carry,
                    scan_xs,
                )
        else:
            # Prepare inputs for scan: time indices
            scan_xs = jnp.arange(num_scan_steps) 

            partial_scan_body = partial(
                _rk_step_scan_body,
                dt=dt,
                f_dys=partial_f_NN,
                f_eI=lambda t_param: jnp.zeros(N * D), # t_param is the time for f_eI
                N=N,
                D=D,
                is_with_dys=isret_dy,
            )
            _, scanned_states_part = lax.scan(
                f=partial_scan_body,
                init=current_block_init_carry,
                xs=scan_xs,
            )
        # Reconstruct the full history for this block
        # scanned_states_part contains states from index 1 up to n_timestep_inblock-1
        states_history_block_computed = jnp.vstack([current_block_init_carry.reshape(1, -1), scanned_states_part])


        # ------------------------ save results ------------------------
        list_ys.append(np.array(states_history_block_computed[:last, : N * D]))

        if isret_dy:
            list_dys.append(np.array(states_history_block_computed[:last, N * D : 2 * N * D]))
        if external_input_dc is not None:
            if isinstance(external_input_dc.soureparam, OUSourceParams):
                # sources_block and eI_block have n_timestep_inblock rows.
                # Slicing with `last` needs to be careful if `last` can be None.
                # If last is None, it means take all rows.
                # If sources_block[last,:] is used, and last is None, it's an error for JAX arrays.
                # However, init_source is used for the *next* block's generation.
                # The state saved (init_states_val) is states_history_block_computed[last, :].
                # So, sources should also correspond to this.
                if last is None: # Corresponds to taking the very last element
                    init_source = sources_block[-1, :] if sources_block.shape[0] > 0 else None
                elif sources_block.shape[0] > 0 : # last is an index like -1 or -2
                    # Ensure last is a valid index for sources_block before slicing
                    actual_last_idx = (sources_block.shape[0] + last) if last < 0 else last
                    if actual_last_idx < sources_block.shape[0] and actual_last_idx >=0:
                        init_source = sources_block[actual_last_idx, :]
                    elif sources_block.shape[0] > 0 : # if last makes it out of bounds, take the actual last
                        init_source = sources_block[-1,:]
                    else: # sources_block is empty
                        init_source = None
                else:
                    raise ValueError("sources_block is empty, cannot set init_source.")
                external_input_dc.soureparam.set_init_source(init_source)

            list_eIs.append(np.array(eI_block[:last, :]))
            list_sources.append(np.array(sources_block[:last]))
        
        # Save the final state of this block to initialize the next one
        if states_history_block_computed.shape[0] > 0:
            if last is None:
                init_states_val = states_history_block_computed[-1, :]
            else:
                # Ensure last is a valid index
                actual_last_idx = (states_history_block_computed.shape[0] + last) if last < 0 else last
                if actual_last_idx < states_history_block_computed.shape[0] and actual_last_idx >=0:
                    init_states_val = states_history_block_computed[actual_last_idx, :]
                elif states_history_block_computed.shape[0] > 0: # if last makes it out of bounds, take the actual last
                     init_states_val = states_history_block_computed[-1,:]
                else: # Should not happen if states_history_block_computed is not empty
                    init_states_val = None 
        else: # Should not happen if n_timestep_inblock >=1
            init_states_val = None


    return TrialResult(
        initial_rndkey=initial_rndkey,
        ys_history=np.vstack(list_ys),
        J=np.asarray(J),
        nd_coeff=np.asarray(nd_coeff),
        hetero_var_vals=np.asarray(hetero_var_vals),
        f_jacob=jax.jacfwd(partial(partial_f_NN, t=0, eI=jnp.zeros(N * D))),
        dys_history=np.vstack(list_dys) if isret_dy and len(list_dys)>0 else None,
        eI_history=np.vstack(list_eIs) if external_input_dc is not None and len(list_eIs)>0 else None,
        eIsource_history=np.vstack(list_sources) if external_input_dc is not None and len(list_sources)>0 else None,
        Win=np.asarray(Win) if Win is not None else None
    ), rndkey

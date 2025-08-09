# if os.environ['JAX_CPU']=='true':
#     print("run JAX with CPU")
#     jax.config.update('jax_platform_name', 'cpu')
# else:
#     print("run JAX with GPU")
#     os.environ['XLA_PYTHON_CLIENT_PREALLOCATE']='false'
#     from jax.config import config; config.update('jax_enable_x64', True)
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jax import lax, grad, vmap
from functools import partial
import matplotlib.pyplot as plt
import tqdm


def get_phi_dot(phi_type):
    if phi_type == "PWL":
        return lambda x: ((x >= -1) * (x <= 1)).astype(jnp.float64)
    elif phi_type == "tanh":
        return vmap(grad(jnp.tanh))
    else:
        raise Exception("not implemented phi dot")


def jacov_2var(xa, w, gammas, mu_gamma_beta, phi_dot=get_phi_dot("PWL"), is_plot=False):
    # mu_gamma_betaはqと同じ
    N = int(xa.size / 2.0)
    x = xa[:N]
    # a = xa[N:]
    phi_dot_x = jnp.repeat(jnp.expand_dims(phi_dot(x), 0), N, 0)
    Jxx = -jnp.eye(N) + jnp.multiply(w, phi_dot_x)
    Jxa = +jnp.eye(N)
    Jax = +mu_gamma_beta * jnp.eye(N)
    Jaa = -jnp.eye(N) @ jnp.diag(gammas)

    Jall = jnp.block([[Jxx, Jxa], [Jax, Jaa]])
    return Jall


def jacov_2var_v2(
    xa, w, gammas, mu_gamma_beta, phi_dot=get_phi_dot("PWL"), is_plot=False
):
    # mu_gamma_betaはqと同じ
    N = int(xa.size / 2.0)
    x = xa[:N]
    # a = xa[N:]
    phi_dot_x = jnp.repeat(jnp.expand_dims(phi_dot(x), 0), N, 0)
    Jxx = -jnp.eye(N) + jnp.multiply(w, phi_dot_x)
    Jxa = +jnp.eye(N)
    Jax = +mu_gamma_beta * jnp.eye(N)
    Jaa = -jnp.eye(N) @ jnp.diag(gammas)

    Jall = jnp.block([[Jxx, Jxa], [Jax, Jaa]])
    return Jall


def jacov_1var(x, w, gammas, phi_dot=get_phi_dot("PWL"), is_plot=False):
    N = x.size
    phi_dot_x = jnp.repeat(jnp.expand_dims(phi_dot(x), 0), N, 0)
    Jxx = -jnp.eye(N) @ jnp.diag(gammas) + jnp.multiply(w, phi_dot_x)

    if is_plot:
        fig = plt.figure()
        ax = plt.subplot(221)
        pcm = plt.imshow(Jxx)
        fig.colorbar(pcm, ax=ax)

    return Jxx


def euler_step(states, params, dt, f):
    new_state = states + dt * f(states, params)
    return new_state


def calc_lyap_from_sim(
    ys,
    rndkey,
    params,
    n_timestep,
    dt,
    norm_steps=None,
    verbose=0,
    is_fori_loop=True,
    mode="2var",
    d0scale=1e-6,
    n_block=50,
):
    """
    y(t)から最大リアプノフ指数を計算する．
    微小摂動d(0)をダイナミクスを線形近似した時の行列Aで時間発展させていく

    \dot{d}(t) = A|_{y=y(t)}d(t)

    最後にd(t)のノルムの時間変化を見ることで最大リアプノフ指数を計算する
    """
    raise Exception("duplicated")

    def norm_d(d, d_amp=0.01):
        scale = jnp.linalg.norm(d) * d_amp
        return d / scale, scale

    def apply_jacov(d, jacov):
        return jacov @ d

    def apply_norm(prev_d, prev_scale):
        scale_factor = jnp.linalg.norm(prev_d)
        new_d = prev_d / scale_factor  # ノルムの大きさが1になるように正規化
        # new_scale = prev_scale*scale_factor
        new_scale = prev_scale + jnp.log10(scale_factor)  # scaleは10^()を意味する
        return new_d, new_scale

    def not_apply_norm(new_d, new_scale):
        return new_d, new_scale

    def one_step(i, d_scale_lambda, ys, dt, f_calcjacov):
        prev_d = d_scale_lambda[i - 1, :-2]
        prev_scale = d_scale_lambda[i - 1, -2]
        new_d = euler_step(prev_d, f_calcjacov(ys[i - 1]), dt, apply_jacov)
        new_d, new_scale = lax.cond(
            jnp.linalg.norm(new_d) > 100, apply_norm, not_apply_norm, new_d, prev_scale
        )

        # new_lambda = 1/(dt*i)*jnp.log(jnp.linalg.norm(new_d)*new_scale/norm_d0)
        new_lambda = jnp.log(
            jnp.linalg.norm(new_d)
            / (jnp.linalg.norm(prev_d))
            * 10 ** (new_scale - prev_scale)
        )
        new_d_scale_lambda = jnp.hstack([new_d, new_scale, new_lambda])
        return d_scale_lambda.at[i].set(new_d_scale_lambda)

    # gamma_dist = params.get("gamma_dist")
    # if gamma_dist.get("dist_type") == "normal":
    #     if norm_steps is None:
    #         norm_steps = int(num_steps/100.)
    #     if mode == "2var":
    #         partial_jacobians_f = partial(jacov_2var, w=params["J"], gammas=params["gammas"], mu_gamma=gamma_dist["mu"], beta=params["beta"], phi_dot=get_phi_dot(params["phi_type"]))
    #     elif mode == "1var":
    #         partial_jacobians_f = partial(jacov_1var, w=params["J"], gammas=params["gammas"], phi_dot=get_phi_dot(params["phi_type"]))

    #     # jacobians = [jit_jacobians_f(xs[i]) for i in range(num_steps)]
    # else: #if gamma_dist.get("dist_type") == "gamma":
    #     if norm_steps is None:
    #         norm_steps = int(num_steps/100.)
    #         if mode == "1var":
    #             partial_jacobians_f = partial(jacov_1var, w=params["J"], gammas=params["gammas"], phi_dot=get_phi_dot(params["phi_type"]))

    d0 = jr.normal(rndkey, shape=[params["N"] * params["D"]]) * d0scale
    rndkey, subkey = jr.split(rndkey)
    D = params["D"]

    if norm_steps is None:
        norm_steps = int(n_timestep / 100.0)
    if D == 2:
        partial_jacobians_f = partial(
            jacov_2var,
            w=params["J"],
            gammas=params["gammas"],
            mu_gamma_beta=params["mu_gamma_beta"],
            phi_dot=get_phi_dot(params["phi_type"]),
        )
    elif D == 1:
        partial_jacobians_f = partial(
            jacov_1var,
            w=params["J"],
            gammas=params["gammas"],
            phi_dot=get_phi_dot(params["phi_type"]),
        )
    else:
        raise Exception()

    # n_blockに分割してシュミレーション

    assert n_timestep % n_block == 0 and n_block >= 1
    list_d_values = []
    list_d_norms = []
    list_scale_values = []
    list_lambda_values = []
    n_timestep_inblock = int(n_timestep / n_block) + 1

    if isinstance(ys, np.ndarray):
        ys = jnp.asarray(ys)

    d_scale_lambda_values = jnp.ones([n_timestep_inblock, d0.size + 2])
    # d_values = jnp.zeros([n_timestep_inblock, d0.size])
    # scale_values = jnp.ones([n_timestep_inblock, 1])
    # lambda_values = jnp.zeros([n_timestep_inblock, 1])
    d_scale_lambda_values = d_scale_lambda_values.at[0].set(
        jnp.concatenate([d0, jnp.ones([1]), jnp.zeros([1])], axis=-1)
    )
    prev_d_scale_lambda_value = None
    for i_block in tqdm.tqdm(range(n_block), desc="lyap block time", leave=False):
        if (
            i_block == 0
        ):  # blockでArrayを流用しつつ，ちゃんとn_timestepになるようにするために，0ブロック目の被りを増やす
            last = -2
        elif i_block == n_block - 1:
            last = None
        else:
            last = -1

        if i_block != 0:
            d_scale_lambda_values = d_scale_lambda_values.at[0].set(
                prev_d_scale_lambda_value
            )

        p_one_step = partial(one_step, ys=ys, dt=dt, f_calcjacov=partial_jacobians_f)

        if is_fori_loop:
            d_scale_lambda_values = lax.fori_loop(
                1, n_timestep_inblock, p_one_step, d_scale_lambda_values
            )
            # d_values = d_scale_lambda_values[:, :-2]
            # scale_values = d_scale_lambda_values[:, -2:-1]
            # lambda_values = d_scale_lambda_values[:, -1:]
        else:
            for i in tqdm.tqdm(range(1, n_timestep_inblock), total=n_timestep_inblock):
                d_scale_lambda_values = p_one_step(i, d_scale_lambda_values)
            # jit_jacobians_f = jit(partial_jacobians_f)
            # scale = 1
            # for i in tqdm.tqdm(range(1, n_timestep_inblock), total=n_timestep_inblock):
            #     d = euler_step(d_values[i-1], jit_jacobians_f(ys[i-1]), dt, apply_jacov)

            #     # 方法１: lambda(t) = 1/t*log(d(t)/d(0))として計算する方法
            #     if i % np.linalg.norm(d) > 100.:
            #         d, newscale = norm_d(d)
            #         scale = scale*newscale
            #     d_values = d_values.at[i].set(d)
            #     scale_values = scale_values.at[i].set(scale)
            #     # lambda_values = lambda_values.at[i].set(1/(dt*i)*jnp.log(jnp.linalg.norm(d)*scale/jnp.linalg.norm(d0)))
            #     lambda_values = lambda_values.at[i].set(jnp.log(jnp.linalg.norm(d)*scale/(jnp.linalg.norm(d_values[i-1])*scale_values[i-1])))

            #     # 方法２: lambda_temp(t) = log(d(t)/d(t-1))として，\sum_{t=0}^T lambda_temp(t)/T = lambda(T)とする方法
            #     # 方法1のlambda(t)と方法2のlambda(T)って同じでは？
        d_values = d_scale_lambda_values[:last, :-2]
        scale_values = d_scale_lambda_values[:last, -2:-1]
        lambda_values = d_scale_lambda_values[:last, -1:]

        prev_d_scale_lambda_value = d_scale_lambda_values[last, :]

        list_d_values.append(np.asarray(d_values))
        list_d_norms.append(
            np.asarray(jnp.expand_dims(jnp.linalg.norm(d_values, axis=-1), -1))
        )
        list_scale_values.append(np.asarray(scale_values))
        list_lambda_values.append(np.asarray(lambda_values))

    # d_values = jnp.vstack(list_d_values)
    scale_values = np.concatenate(list_scale_values, axis=0)
    d_norms = np.concatenate(list_d_norms, axis=0)
    scaled_d_norms = np.multiply(d_norms, 10**scale_values)
    lambda_values = np.concatenate(list_lambda_values, axis=0) / dt

    mean_lambda_values = np.cumsum(lambda_values) / (
        np.arange(1, lambda_values.size + 1)
    )  # lambda_{mean}(Nstep) = 1/(Nstep)\sum_{i=1}^{Nstep}lambda(i)

    return scaled_d_norms, lambda_values, mean_lambda_values, rndkey


def calc_lyap_from_sim_v2(
    ys,
    rndkey,
    params,
    n_timestep,
    dt,
    f_jacob=None,
    norm_steps=100,
    verbose=0,
    is_fori_loop=True,
    n_block=50,
):
    """
    y(t)から最大リアプノフ指数を計算する．
    微小摂動d(0)をダイナミクスを線形近似した時の行列Aで時間発展させていく

    \dot{d}(t) = A|_{y=y(t)}d(t)

    最後にd(t)のノルムの時間変化を見ることで最大リアプノフ指数を計算する

    \lambda(t) := log(||d(t)||/||d(t-1)||)

    最終的なリアプノフ指数を 1/T \sum lambda(t)で計算してもいいし,
    lambda(t)を陽に計算せずにscale(T) - scale(0) + \ln |d(T)| - \ln |d(0)|で計算してもいい (Notion参照)
    """

    def apply_jacov(d, jacov):
        return jacov @ d

    def apply_norm(prev_d, prev_scale):
        scale_factor = jnp.linalg.norm(prev_d)
        new_d = prev_d / scale_factor  # ノルムの大きさが1になるように正規化
        new_scale = prev_scale + jnp.log(scale_factor)  # scaleはe^()を意味する
        return new_d, new_scale

    def not_apply_norm(new_d, new_scale):
        return new_d, new_scale

    def one_step(i, d_scale_lambda, ys, dt, f_calcjacov):
        prev_d = d_scale_lambda[i - 1, :-2]
        prev_scale = d_scale_lambda[i - 1, -2]
        new_d = euler_step(prev_d, f_calcjacov(ys[i - 1]), dt, apply_jacov)
        new_d, new_scale = lax.cond(
            i % norm_steps == 0, apply_norm, not_apply_norm, new_d, prev_scale
        )  # ノルムが100を超えると正規化

        new_lambda = (
            jnp.log(jnp.linalg.norm(new_d) / (jnp.linalg.norm(prev_d)))
            + new_scale
            - prev_scale
        )
        new_d_scale_lambda = jnp.hstack([new_d, new_scale, new_lambda])
        return d_scale_lambda.at[i].set(new_d_scale_lambda)

    def get_eigvec_largest_eigval(mat):
        eigenvalues, eigenvectors = np.linalg.eig(np.asarray(mat))
        return jnp.asarray(np.real(eigenvectors[:, np.argmax(np.real(eigenvalues))]))

    d0_inittype = params["d0_init"]["type"]
    d0scale = params["d0_init"]["scale"]
    if d0_inittype == "randnormal":
        d0 = jr.normal(
            rndkey, shape=[params["N"] * params["D"]]
        )  # 初期値は正規分布からランダムにとる
        rndkey, subkey = jr.split(rndkey)
    elif d0_inittype == "value":
        d0_value = params["d0_init"]["value"]
        d0 = jnp.asarray(d0_value)
    elif d0_inittype == "jacob_origin":
        jacob_mat = f_jacob(jnp.zeros(shape=[params["N"] * params["D"]]))
        d0 = get_eigvec_largest_eigval(jacob_mat)
    elif d0_inittype == "jacob_y0":
        jacob_mat = f_jacob(jnp.asarray(ys[0]))
        d0 = get_eigvec_largest_eigval(jacob_mat)
    elif d0_inittype == "jacob_yt":
        n_step = params["d0_init"]["n_step"]
        jacob = jnp.eye(ys[0].size)
        for i in range(n_step):
            jacob = jnp.matmul(
                jnp.eye(ys[0].size) + f_jacob(jnp.asarray(ys[i])) * dt, jacob
            )
            if i % 100 == 0:
                jacob = jacob / jnp.linalg.norm(jacob)
        d0 = get_eigvec_largest_eigval(jacob)
    else:
        raise Exception("invalid initialization mode")
    d0 = d0 / jnp.linalg.norm(d0) * d0scale
    D = params["D"]

    if norm_steps is None:
        norm_steps = int(n_timestep / 100.0)

    if f_jacob is None:
        if D == 2:
            partial_jacobians_f = partial(
                jacov_2var,
                w=params["J"],
                gammas=params["hetero_vals"]["gamma"],
                mu_gamma_beta=params["net_params"]["q"],
                phi_dot=get_phi_dot(params["phi_type"]),
            )
        elif D == 1:
            partial_jacobians_f = partial(
                jacov_1var,
                w=params["J"],
                gammas=params["hetero_vals"]["gamma"],
                phi_dot=get_phi_dot(params["phi_type"]),
            )
        else:
            raise Exception()
    else:
        partial_jacobians_f = f_jacob

    # n_blockに分割してシュミレーション

    assert n_timestep % n_block == 0 and n_block >= 1
    list_d_values = []
    list_d_norms = []
    list_scale_values = []
    list_lambda_values = []
    n_timestep_inblock = int(n_timestep / n_block) + 1

    if isinstance(ys, np.ndarray):
        ys = jnp.asarray(ys)

    d_scale_lambda_values = jnp.zeros(
        [n_timestep_inblock, d0.size + 2]
    )  # scaled_d(t)とscale(t)とlambda(t)
    d_scale_lambda_values = d_scale_lambda_values.at[0].set(
        jnp.concatenate([d0, jnp.zeros([1]), jnp.zeros([1])], axis=-1)
    )  # scaled_d(0)=d(0), scale(0)=0, lambda(0)=0
    prev_d_scale_lambda_value = None
    for i_block in tqdm.tqdm(range(n_block), desc="lyap block time", leave=False):
        if (
            i_block == 0
        ):  # blockでArrayを流用しつつ，ちゃんとn_timestepになるようにするために，0ブロック目の被りを増やす
            last = -2
        elif i_block == n_block - 1:
            last = None
        else:
            last = -1

        if i_block != 0:
            d_scale_lambda_values = d_scale_lambda_values.at[0].set(
                prev_d_scale_lambda_value
            )

        p_one_step = partial(one_step, ys=ys, dt=dt, f_calcjacov=partial_jacobians_f)

        if is_fori_loop:
            d_scale_lambda_values = lax.fori_loop(
                1, n_timestep_inblock, p_one_step, d_scale_lambda_values
            )
        else:
            for i in tqdm.tqdm(range(1, n_timestep_inblock), total=n_timestep_inblock):
                d_scale_lambda_values = p_one_step(i, d_scale_lambda_values)

        d_values = d_scale_lambda_values[:last, :-2]
        scale_values = d_scale_lambda_values[:last, -2:-1]
        lambda_values = d_scale_lambda_values[:last, -1:]

        prev_d_scale_lambda_value = d_scale_lambda_values[last, :]

        list_d_values.append(np.asarray(d_values))
        list_d_norms.append(
            np.asarray(jnp.expand_dims(jnp.linalg.norm(d_values, axis=-1), -1))
        )
        list_scale_values.append(np.asarray(scale_values))
        list_lambda_values.append(np.asarray(lambda_values))

    # d_values = jnp.vstack(list_d_values)
    scale_values = np.concatenate(list_scale_values, axis=0)
    d_norms = np.concatenate(list_d_norms, axis=0)
    scaled_d_norms = np.multiply(d_norms, np.exp(scale_values))
    lambda_values = np.concatenate(list_lambda_values, axis=0)[1:]

    cumsum_lambda_values = np.cumsum(lambda_values) / (
        np.arange(1, lambda_values.size + 1) * dt
    )

    # lambda(t)を逐次陽に計算せずに最大リアプノフ指数計算するやりかた, どっちでもいい
    # cumsum_lambda_values_2 = (scale_values[:,0] - scale_values[0] + np.log(d_norms[:,0]) - np.log(d_norms[0])) /(np.arange(1, lambda_values.size+1)*dt)
    # print(np.mean(cumsum_lambda_values-cumsum_lambda_values_2))

    return d_values[-1, :], scaled_d_norms, lambda_values, cumsum_lambda_values, rndkey

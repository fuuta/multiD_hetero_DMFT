import os
import random
from .logging_utils import get_logger

logger = get_logger()

cuda_device_pool = ["3"]


def set_jax_device_config(cuda_device: str | None = "auto") -> None:
    import jax

    # jaxでGPUを使うかのセッティング
    if os.environ.get("JAX_CPU") == "true" or cuda_device is None:
        logger.info("run JAX with CPU")
        jax.config.update("jax_platform_name", "cpu")
    else:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        cuda_device_now = os.environ.get("CUDA_VISIBLE_DEVICES")
        if cuda_device_now is None:
            if cuda_device == "auto":
                cuda_device = random.choice(cuda_device_pool)
            os.environ["CUDA_VISIBLE_DEVICES"] = (
                cuda_device  # CUDAデバイスの決め打ち (nvidia-smiで表示されるGPUの2番目を使う)
            )
        cuda_device = os.environ.get("CUDA_VISIBLE_DEVICES")
        # GPUメモリは必要な時に必要な分だけ確保するようにする
        if os.environ.get("XLA_PYTHON_CLIENT_PREALLOCATE") is None:
            os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
        jax.config.update("jax_enable_x64", True)
        logger.info("run JAX with GPU on device %s", cuda_device)


def check_jax_device() -> str:
    import jax

    logger.info("JAX device: %s", jax.devices())

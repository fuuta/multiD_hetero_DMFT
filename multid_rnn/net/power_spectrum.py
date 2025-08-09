import numpy as onp
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from jax.scipy.special import erf


def julia_range(t_min, d, t_max):
    return jnp.arange(t_min, t_max + d, d)
    # return jnp.linspace(t_min, t_max, int((t_max-t_min)/d)+1)


def gettrangecorr(freqRange):
    dFreq = freqRange[1] - freqRange[0]
    freqMax = jnp.abs(freqRange[0])
    dt = 1 / (2 * freqMax)
    tMax = 0.5 * dt * freqRange.size
    # return  -0.5 ./dFreq:1 ./(2*freqMax):0.5 ./dFreq
    return julia_range(-tMax, dt, (tMax - dt))


def gpaFilter(freq, gamma, beta, tauX=1, verbose=0):
    if verbose >= 1:
        print("\t use gpaFilter")
    omegas = 2 * jnp.pi * freq
    if gamma > 0:  # これtauX=1を仮定した計算じゃね？
        return (gamma**2 + omegas**2) / (
            (1 + gamma) ** 2 * omegas**2 + (gamma - gamma * beta - omegas**2) ** 2
        )
    elif gamma == 0:
        return 1 / (1 + tauX**2 * omegas**2)


def gpafilterhetero_v2(freq, meanGamma, beta, sigmaGamma, verbose=0):
    if verbose >= 1:
        print("\t use gpafilterhetero v2")
    omega = 2 * jnp.pi * freq
    rho = meanGamma * beta
    chiSq = gpaFilter(freq, gamma=meanGamma, beta=beta, verbose=0)
    ret = rho**2 * sigmaGamma**2
    ret = ret / (
        (
            (meanGamma + 1) ** 2 * omega**2
            - (omega**2 + 1) * sigmaGamma**2
            + (omega**2 - meanGamma + rho) ** 2
        )
        * (meanGamma**2 + omega**2)
    )
    ret = ret + 1
    ret = ret * chiSq
    # if verbose >= 1
    #     println("meanGamma", meanGamma, ", beta", beta, ", sigmaGamma", sigmaGamma)
    #     println("\t filter ratio(f=0): ", ret[Int(floor(size(freq, 1)/2.))]/chiSq[Int(floor(size(freq, 1)/2.))])
    # end
    return ret


def PWLnonlinearpass(S, gF, dFreq=0.001, maxFreq=0.5, dIntSigmaFactor=100, verbose=0):
    """
    生信号のパワースペクトルから活性化関数適用後信号のパワースペクトルを計算する関数

    区分線形(PWL)活性化関数を使うことで, 生信号のパワースペクトルとG(f)の関数の2重積分で活性化関数適用後信号のパワースペクトルが計算できる
    内部では,
    1. パワースペクトルF[x(t)](\tau)に対応した時系列信号\tilde x(t)の生成
    2. \phi(\tilde x(t))の計算
    3. 最後にパワースペクトルに再変換F[\phi(\tilde x(t))](\tau)

    1. 生信号のパワースペクトル S_x(f)=F[x(t)](\tau) をフーリエ逆変換し, 生信号の自己相関関数 C_x(\tau) を計算
    2. 生信号の自己相関関数 C_x から 活性化関数適用後信号のパワースペクトル C_ϕ(x) を計算 (教科書の手法のPWL版, Priceの定理を用いる)
    3. 活性化関数適用後信号の自己相関関数 C_ϕ(x) を フーリエ変換して 活性化関数適用後信号のパワースペクトル S_ϕ(x)を計算

    S: 生信号のパワースペクトル S_x(f)
    gF: 活性化関数
    dFreq: 周波数空間の刻み幅,
    maxFreq: 周波数空間の最大, 最小値
    dIntSigmaFactor: 数値積分の刻み幅
    """
    if verbose >= 2:
        print(
            "use PWLnonlinearpass (区分求積), dIntSigmaFactor={}".format(
                dIntSigmaFactor
            )
        )

    freqRange = julia_range(-maxFreq, dFreq, maxFreq)

    dt = 1.0 / (2.0 * maxFreq)
    cX = (
        dFreq * S.size * jnp.fft.ifft(jnp.fft.ifftshift(S))
    )  # パワースペクトルをifftするー＞自己相関関数に，ifftシフトしてるので，正の周波数の小から大の後に負の周波数小から大の配列
    index0 = 0  # the index that indicate the autocorrelation at 0-time lag
    cX0 = cX[index0].real  # c(τ=0)の値
    dIntSigma = (
        cX0 / dIntSigmaFactor
    )  # dIntSigmaはσに関する積分を数値的に行う時のdσの刻み幅，たぶんdσ'も同じ刻み幅
    cPhi = onp.zeros_like(cX)

    # tmpFun = s -> 1 ./sqrt(1-s^2 ./(cX0^2)) * exp(-1 ./(cX0*(1-s^2 ./(cX0.^2) ))) * sinh(s./(cX0^2*(1-s^2 ./(cX0^2) )))
    # tmpFunは二重積分の中身，s=σ, sinh(x)=(exp(x)-exp(-x))/2の形でexp formにする
    def tmpFun(s):
        ret = 1.0 / 2.0 * 1.0 / jnp.sqrt(1 - s**2 / (cX0**2))
        ret = ret * (
            jnp.exp(-(1.0 - s / cX0) / (cX0 * (1 - s**2.0 / (cX0**2))))
            - jnp.exp(-(1.0 + s / cX0) / (cX0 * (1 - s**2.0 / (cX0**2))))
        )
        return ret

    for i in range(cX.size):  # 各時間遅れに対してc_x(τ)をc_ϕ(x)(τ)に変換する
        if cX[i] >= 0:
            cPhi[i] = (
                erf(1.0 / jnp.sqrt(2 * cX0)) ** 2 * cX[i]
                + 2.0 / (jnp.pi * cX0) * (dIntSigma) ** 2
            )
            cPhi[i] *= jnp.sum(
                jax.lax.map(
                    lambda sigmaP: jnp.sum(tmpFun(julia_range(0.0, dIntSigma, sigmaP))),
                    julia_range(0, dIntSigma, jnp.max(cX[i] - dIntSigma, dIntSigma)),
                )
            )  # sum2回*(dσ)^2 することで2重積分を数値計算してる．
        # elseif cX[i]<0
        #     cPhi[i] = erf(1 ./sqrt(2*cX0))^2 * cX[i] - 2 ./(pi*cX0).* (dIntSigma).^2 .*
        #     sum(map( sigmaP->sum( tmpFun.(0.:dIntSigma:sigmaP) ),
        #             min(cX[i]+dIntSigma,-dIntSigma):dIntSigma:0.))
        else:  # cが負になることとかあり得るの・・・？
            cPhi[i] = (
                erf(1.0 / jnp.sqrt(2 * cX0)) ** 2 * cX[i]
                - 2.0 / (jnp.pi * cX0) * (dIntSigma) ** 2
            )
            cPhi[i] *= jnp.sum(
                jax.lax.map(
                    lambda sigmaP: jnp.sum(tmpFun(julia_range(0.0, dIntSigma, sigmaP))),
                    julia_range(0, dIntSigma, jnp.max(-cX[i] - dIntSigma, dIntSigma)),
                )
            )

    # cReconstructed = copy(cX);
    # cReconstructed[round(Int,0.5*size(cX,1))+1:end] = cReconstructed[round(Int,0.5*size(cX,1))+1:-1:1]
    sPhi = dt * jnp.fft.fftshift(jnp.fft.fft(cPhi))  # 周波数の関数に変換
    return sPhi


def sampleFTfromspectrum2(freqRange, S, method="realImag", shift=True):
    """
    ※ freqRangeは0を中心として負から正になる配列で奇数サイズであるとする

    """
    if method == "realImag":
        """
        アイディアとしては, 
        生成するFT信号にsqrt(1/2*S)に標準正規分布をかけることで,
        パワースペクトルに再変換すると(共役とかけたときに), 分散が1になってもとに戻る 
        ※ パワースペクトル空間で平均をとればもとに戻るだけで, 1サンプルではもとには戻らないよ (notes/2024-02-22/test_powerspectrum.ipynb 参照)
        """

        # tMax = 0.5*(size(freqRange,1)-1)
        maxFreq = freqRange[-1]
        dFreq = freqRange[1] - freqRange[0]
        floor_size = jnp.floor(freqRange.size / 2.0)

        # tMax = 2 .*maxFreq./dFreq ### 1 ./dt * t_max
        tMax = (freqRange.size * dFreq) * (
            freqRange.size - 1
        )  ### 1 ./dt * t_max -- to understand why the factor in front
        HS = jnp.where(freqRange == 0)[0]  # 0の位置決定

        ### tMaxhe alternative option, maybe more correct, is to sample the real and imaginary parts, instead of the
        ### amplitude and phase.

        S = jnp.maximum(
            jnp.zeros_like(S), jnp.real(S)
        )  # Sはパワースペクトルなので, 実部が負はありえないので0とする
        if shift:
            # realPart = sqrt.(tMax.*0.5 .*S[2:HS]).*randn(HS-1)
            # imagPart = sqrt.(tMax.*0.5 .*S[2:HS]).*randn(HS-1)
            # FT = vcat(sqrt.(tMax.*S[1]).*randn(), realPart+im.*imagPart, sqrt.(tMax.*S[HS+1]).*randn(), (realPart-im.*imagPart)[end:-1:1])
            realPart = jnp.sqrt(tMax * 0.5 * S[:floor_size]) * jnp.random.normal(
                size=floor_size
            )
            imagPart = jnp.sqrt(tMax * 0.5 * S[:floor_size]) * jnp.random.normal(
                size=floor_size
            )
            FT = jnp.vstack(
                [
                    realPart + 1j * imagPart,
                    jnp.sqrt(tMax * S[HS])
                    * jnp.random.normal(),  # F[x](\tau=0)に対応する点だけ虚部が0なのはなんでだ？
                    (realPart - 1j * imagPart)[::1],
                ]
            )
            # FT = vcat(realPart+im.*imagPart, sqrt.(0.5 .*tMax.*S[HS]).*randn(), (realPart-im.*imagPart)[end:-1:1], sqrt.(0.5 .*tMax.*S[end]).*randn())

        else:
            raise Exception
    else:
        raise Exception
    return FT


def sampleFTfromspectrum_realImag(freqRange, S, HS, floor_size, shift=True):
    """
    ※ freqRangeは0を中心として負から正になる配列で奇数サイズであるとする


    アイディアとしては,
    生成するFT信号にsqrt(1/2*S)に標準正規分布をかけることで,
    パワースペクトルに再変換すると(共役とかけたときに), 分散が1になってもとに戻る
    """

    # tMax = 0.5*(size(freqRange,1)-1)
    maxFreq = freqRange[-1]
    dFreq = freqRange[1] - freqRange[0]

    # tMax = 2 .*maxFreq./dFreq ### 1 ./dt * t_max
    tMax = (freqRange.size * dFreq) * (
        freqRange.size - 1
    )  ### 1 ./dt * t_max -- to understand why the factor in front
    tMax = 1  # ここ1で良くね？？？？

    ### tMaxhe alternative option, maybe more correct, is to sample the real and imaginary parts, instead of the
    ### amplitude and phase.

    S = jnp.maximum(
        jnp.zeros_like(S), jnp.real(S)
    )  # Sはパワースペクトルなので, 実部が負はありえないので0とする
    if shift:
        # realPart = sqrt.(tMax.*0.5 .*S[2:HS]).*randn(HS-1)
        # imagPart = sqrt.(tMax.*0.5 .*S[2:HS]).*randn(HS-1)
        # FT = vcat(sqrt.(tMax.*S[1]).*randn(), realPart+im.*imagPart, sqrt.(tMax.*S[HS+1]).*randn(), (realPart-im.*imagPart)[end:-1:1])
        realPart = jnp.sqrt(tMax * 0.5 * S[:floor_size]) * jnp.random.normal(
            size=floor_size
        )
        imagPart = jnp.sqrt(tMax * 0.5 * S[:floor_size]) * jnp.random.normal(
            size=floor_size
        )
        FT = jnp.concatenate(
            [
                realPart + 1j * imagPart,
                jnp.sqrt(tMax * S[HS])
                * jnp.random.normal(
                    size=[1]
                ),  # F[x](\tau=0)に対応する点だけ虚部が0なのはなんでだ？
                (realPart - 1j * imagPart)[::-1],
            ]
        )
        # FT = vcat(realPart+im.*imagPart, sqrt.(0.5 .*tMax.*S[HS]).*randn(), (realPart-im.*imagPart)[end:-1:1], sqrt.(0.5 .*tMax.*S[end]).*randn())

    else:
        raise Exception
    return FT


def numericalnonlinearpass(
    S,
    gF,
    dFreq=0.0002,
    maxFreq=1.0,
    M=100,
    method="realImag",
    withCosDrive=False,
    cosAmp=[],
    cosFreq=[],
    shift=True,
    center=False,
    substractMean=True,
):
    if shift:
        freqRange = julia_range(-maxFreq, dFreq, maxFreq)
    else:
        freqRange = jnp.vstack(
            [
                julia_range(0, dFreq, maxFreq),
                julia_range(-(maxFreq - dFreq), dFreq, -dFreq),
            ]
        )
    assert freqRange.size % 2 == 1

    tMax = 1 / dFreq
    dt = 1 / (2 * maxFreq)
    T = tMax / dt
    tRange = gettrangecorr(freqRange)
    # FTx = Array{Complex{Float64},2}(zeros(M,size(freqRange,1)))
    # FTx = Array{Complex{Float64},2}(zeros(size(freqRange,1)))
    SxPhi = jnp.zeros(S.size)
    meanPhi = 0.0

    for i in range(M):
        FTx = sampleFTfromspectrum2(freqRange, S, method=method, shift=shift)

        if shift:
            x = jnp.real(jnp.fft.ifftshift(jnp.fft.ifft(jnp.fft.ifftshift(FTx))))
            if withCosDrive:
                x = x + cosAmp * jnp.cos(
                    2 * jnp.pi * cosFreq * tRange + 2 * jnp.pi * jnp.random.normal()
                )

            # apply gain
            phiX = gF(x)
            if center:
                phiX = phiX - jnp.mean(phiX)

            # transform back
            phiFT = jnp.fft.fftshift(jnp.fft.fft(phiX))
        else:
            x = jnp.real(jnp.fft.ifft(FTx, 2))
            if withCosDrive:
                x = x + cosAmp * jnp.cos(
                    2 * jnp.pi * cosFreq * tRange + 2 * jnp.pi * jnp.random.normal()
                )

            phiX = gF(x)
            if center:
                phiX = phiX - jnp.mean(phiX)

            phiFT = jnp.fft.fftshift(jnp.fft.fft(phiX))

        # average to get the spectrum of phi

        meanPhi = (i - 1) / i * meanPhi + 1 / i * jnp.mean(phiX)
        # SxPhi = (i-1)./i .* SxPhi +
        #    1 ./i .*2 ./(T) .*conj.(phiFT).*phiFT ### version with the factor 2
        # SxPhi = (i-1)./i .* SxPhi +
        #     1 ./i .*1 ./(T) .*conj.(phiFT).*phiFT ### version without the factor 2
        SxPhi = (i - 1) / i * SxPhi + 1 / i * 1 / (
            (freqRange.size * dFreq) * T
        ) * jnp.conj(
            phiFT
        ) * phiFT  ### version without the factor , but with a weird factor in front

    if substractMean:
        # SxPhi[find(x->x==0,freqRange)] += -(meanPhi^2)./(dFreq)
        pass

    return SxPhi


def iterativemethod(
    iterations,
    freqRange,
    S0,
    g,
    gF,
    nonlinearpass,
    dFreq,
    maxFreq,
    saveAll=False,
    netType="adapt",
    tauA=100.0,
    beta=1.0,
    tauX=1.0,
    externalInput=False,
    externalSpectrum=[],
    stopAtConvergence=False,
    verbose=False,
    sigmaBeta=0.0,
    sigmaGamma=0.0,
    withCosDrive=False,
    distanceTh=1e-2,
    **kwargs,
):
    if saveAll:
        pass
    else:
        Sx = jnp.copy(S0)
        if netType == "adapt":
            pass
            # F = adaptFilter(freqRange; tauX=tauX,tauA=tauA,beta=beta)
        # elif netType=="standard":
        #     F = standardfilter(freqRange; tauX=tauX)
        # elif netType=="gpa":
        #     F = gpaFilter(freqRange; tauX=tauX,tauA=tauA,beta=beta)
        # elif netType=="heteroadapt":
        #     # temp_f = x -> adaptfilterhetero(x; gamma=1 ./ tauA, meanBeta=beta, sigmaBeta=sigmaBeta)
        #     # F = map(temp_f, freqRange)
        #     F = adaptfilterhetero2(freqRange; gamma=1 ./ tauA, meanBeta=beta, sigmaBeta=sigmaBeta)
        # elif netType=="heterogpa":
        #     F = gpafilterhetero(freqRange; meanGamma=1 ./ tauA, beta=beta, sigmaGamma=sigmaGamma)
        #     # println(mean(F .* g^2))
        #     # throw(ErrorException("hoge"))
        elif netType == "heterogpa_v2":
            # println(1 ./ tauA, " ", beta, " ", sigmaGamma)
            F = gpafilterhetero_v2(
                freqRange, meanGamma=1.0 / tauA, beta=beta, sigmaGamma=sigmaGamma
            )
            # println(mean(F .* g^2))
            # throw(ErrorException("hoge"))
            # println("\t\t min F: ", minimum(F), " omega:", freqRange[argmin(F)]*2*pi)
            # println("\t\t max F: ", maximum(F), " omega:", freqRange[argmax(F)]*2*pi)
            # println("\t\t maxF*g^2: ", maximum(F)*g^2)
            # F = gpafilterhetero(freqRange; meanGamma=1 ./ tauA, beta=beta, sigmaGamma=sigmaGamma)
            # exit()
            # print(F)
            plt.plot(freqRange, F)
            plt.grid()

        # println("\t\t min F: ", minimum(F), " omega:", freqRange[argmin(F)]*2*pi)
        # println("\t\t max F: ", maximum(F), " omega:", freqRange[argmax(F)]*2*pi)
        # println("\t\t maxF*g^2: ", maximum(F)*g^2)

        # max_f_index = jnp.argmax(F)

        #

        for iter in range(iterations):
            SxPhi = nonlinearpass(Sx, gF, dFreq=dFreq, maxFreq=maxFreq, **kwargs)

            #
            # scale with the gain and linear filtering
            if externalInput:
                Sx = F * (g**2 * SxPhi + externalSpectrum)
            else:
                Sx = F * g**2 * SxPhi

        # println("\t max Sx(f): ", maximum(real(Sx)), ", argmax_f Sx(f): ", freqRange[argmax(real(Sx))], ", argmax_f G(f): ", freqRange[max_f_index])
        return Sx

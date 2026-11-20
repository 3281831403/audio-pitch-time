"""播客变调不变速核心：Hann 分帧、基频周期估计、WSOLA 拉伸与批量处理。

约定
----
* 样本：单声道 ``list[float]``，标称取值范围 ``-1.0 .. 1.0``；含
  ``NaN``、``inf`` 或非数值元素时视为非法音频。
* 采样率：需要采样率的函数由调用方传入整数 Hz（例如 ``16000``）；
  所有长度与 hop 均以采样点为单位，不按秒换算。
* 帧长 / hop：``frame_len`` 是每帧采样数，``hop`` 是相邻帧起点的间隔。
  分帧、拉伸与重叠相加都要求 ``frame_len > hop >= 1``：hop 为 0 会除零，
  ``frame_len <= hop`` 时窗无法正常重叠相加，必须抛 :class:`StretchError`。
* 窗：对称 Hann 窗
  ``w[n] = 0.5 - 0.5 * cos(2 * pi * n / (frame_len - 1))``，
  其中 ``0 <= n < frame_len``。
* 容差：浮点比较统一使用绝对容差 ``1e-6``。
* 异常：参数、样本或批量输入非法时统一抛 :class:`StretchError`。
"""

from __future__ import annotations

import math


TOLERANCE = 1e-6
DEFAULT_SAMPLE_RATE = 16000
DEFAULT_FRAME_LEN = 1024
DEFAULT_HOP = 256

_MIN_FRAME_LEN = 2
_MAX_FRAME_LEN = 65536
_MIN_RATE = 0.1
_MAX_RATE = 8.0
_MAX_SEMITONES = 48.0

# Hann 窗和低于该阈值视为边界零区，直接输出 0.0，避免除零。
_WINDOW_EPS = 1e-12


class StretchError(ValueError):
    """参数、音频样本或批量请求非法时抛出的异常。"""


# ---------------------------------------------------------------------------
# 内部工具
# ---------------------------------------------------------------------------


def _is_real_int(value: object) -> bool:
    """严格判断“真正的整数”：``bool`` 不算整数。"""
    return isinstance(value, int) and not isinstance(value, bool)


def _hann_window(frame_len: int) -> list[float]:
    """对称 Hann 窗 ``w[n] = 0.5 - 0.5*cos(2*pi*n/(frame_len-1))``。"""
    if frame_len <= 1:
        return [1.0] * frame_len
    denom = float(frame_len - 1)
    return [
        0.5 - 0.5 * math.cos(2.0 * math.pi * n / denom)
        for n in range(frame_len)
    ]


def _validate_samples(samples: object) -> list[float]:
    """校验单声道样本序列：必须是 list，元素必须为有限数值。"""
    if not isinstance(samples, list):
        raise StretchError("samples 必须是 list[float]")
    for value in samples:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise StretchError("样本必须是数值类型")
        if not math.isfinite(float(value)):
            raise StretchError("样本包含 NaN 或 inf")
    return samples


def _validate_frame_hop(frame_len: object, hop: object) -> None:
    """校验帧长 / hop：真整数、``frame_len > hop >= 1``、帧长有界。"""
    if not _is_real_int(frame_len) or not _is_real_int(hop):
        raise StretchError("frame_len 与 hop 必须是整数")
    if frame_len < _MIN_FRAME_LEN or frame_len > _MAX_FRAME_LEN:
        raise StretchError(
            f"frame_len 必须在 {_MIN_FRAME_LEN}..{_MAX_FRAME_LEN} 之间"
        )
    if hop < 1:
        raise StretchError("hop 必须 >= 1，否则会除零")
    if frame_len <= hop:
        raise StretchError("frame_len 必须大于 hop，否则无法重叠相加")


def _validate_sample_rate(sample_rate: object) -> None:
    if not _is_real_int(sample_rate) or sample_rate <= 0:
        raise StretchError("sample_rate 必须是正整数 Hz")


def _frame_params_for_rate(sample_rate: int) -> tuple[int, int]:
    """按采样率折算约 64 ms 帧长、75% 重叠的 hop（16 kHz 下 1024/256）。"""
    frame_len = int(round(sample_rate * 0.064))
    frame_len = max(_MIN_FRAME_LEN, frame_len)
    # 保证 hop < frame_len。
    hop = max(1, frame_len // 4)
    return frame_len, hop


def _linear_resample(samples: list[float], target_len: int) -> list[float]:
    """线性插值重采样到精确的 ``target_len`` 个采样（长度按采样点取整）。"""
    source_len = len(samples)
    if target_len <= 0:
        return []
    if source_len == 1:
        return [samples[0]] * target_len
    if target_len == 1:
        return [samples[0]]
    out: list[float] = []
    scale = (source_len - 1) / (target_len - 1)
    for k in range(target_len):
        pos = k * scale
        idx = int(math.floor(pos))
        frac = pos - idx
        if idx >= source_len - 1:
            out.append(samples[source_len - 1])
        else:
            out.append(
                samples[idx] * (1.0 - frac) + samples[idx + 1] * frac
            )
    return out


def _moving_average_envelope(samples: list[float], window: int) -> list[float]:
    """瞬时幅度包络（|x| 的滑动平均），用于共振峰强度估计。"""
    n = len(samples)
    if n == 0:
        return []
    radius = max(1, window // 2)
    envelope: list[float] = []
    for i in range(n):
        lo = max(0, i - radius)
        hi = min(n, i + radius + 1)
        seg = samples[lo:hi]
        envelope.append(sum(abs(v) for v in seg) / len(seg))
    return envelope


# ---------------------------------------------------------------------------
# 公开 API
# ---------------------------------------------------------------------------


def frame_signal(
    samples: list[float],
    frame_len: int,
    hop: int,
) -> list[list[float]]:
    """把单声道样本切成互相重叠的 Hann 加窗帧。

    契约
    ----
    * ``samples`` 为 ``list[float]``；空输入返回 ``[]``。
    * ``frame_len`` 为 ``>= 2`` 的整数，``hop`` 为 ``>= 1`` 的整数，且必须
      ``frame_len > hop``；否则抛 :class:`StretchError`（hop 为 0 或帧长
      不大于 hop 会造成除零/无法重叠）。
    * 只输出完整帧，**不补零**：第 ``k`` 帧覆盖
      ``samples[k * hop : k * hop + frame_len]``；当
      ``len(samples) < frame_len`` 时返回 ``[]``，否则帧数为
      ``(len(samples) - frame_len) // hop + 1``。
    * 每帧乘对称 Hann 窗
      ``w[n] = 0.5 - 0.5 * cos(2 * pi * n / (frame_len - 1))``；
      因此帧首尾采样被压到 0，中间保留原幅度。
    * 采样率不参与本函数（帧长与 hop 已经以采样点计）；浮点比较容差
      ``1e-6``。
    """
    _validate_samples(samples)
    _validate_frame_hop(frame_len, hop)

    n = len(samples)
    if n < frame_len:
        # 不补零：凑不满一帧就不输出，避免边界爆音。
        return []

    window = _hann_window(frame_len)
    frame_count = (n - frame_len) // hop + 1
    frames: list[list[float]] = []
    for k in range(frame_count):
        start = k * hop
        frame = [
            samples[start + j] * window[j] for j in range(frame_len)
        ]
        frames.append(frame)
    return frames


def detect_period(
    frames: list[list[float]],
    min_lag: int,
    max_lag: int,
) -> list[int]:
    """用归一化自相关估计每帧的基频周期（单位：采样点）。

    契约
    ----
    * ``frames`` 为帧序列；空输入返回 ``[]``。
    * 搜索范围 ``min_lag .. max_lag``（含端点，整数）先钳制到
      ``[2, len(frame) - 1]``；越界输入本身不报错，只做钳制。钳制后下界
      大于上界时抛 :class:`StretchError`。
    * 对每帧计算归一化自相关
      ``r(tau) = sum(x[i] * x[i+tau]) / sqrt(sum(x[i]**2) * sum(x[i+tau]**2))``，
      分母为 0 时取 ``r(tau) = 0``；不得使用未归一化的长 lag 累加值。
    * 取 ``r(tau)`` 最大处的 lag；若多个 lag 与最大值相差不超过 ``1e-9``，
      取最小 lag（周期 80 的帧不会输给 160 的长 lag）。
    * 返回与 ``frames`` 等长的 ``list[int]``；采样率与 hop 都不参与计算
      （帧已经分好），周期以采样点计，浮点比较容差 ``1e-6``。
    """
    if not isinstance(frames, list):
        raise StretchError("frames 必须是 list")
    if not frames:
        return []
    if not _is_real_int(min_lag) or not _is_real_int(max_lag):
        raise StretchError("min_lag 与 max_lag 必须是整数")

    periods: list[int] = []
    for frame in frames:
        if not isinstance(frame, list):
            raise StretchError("每一帧都必须是 list[float]")
        frame_len = len(frame)
        low = max(2, min_lag)
        high = min(frame_len - 1, max_lag)
        if low > high:
            raise StretchError("钳制后 min_lag > max_lag，无法搜索周期")

        # 第一遍：计算全部候选 lag 的归一化自相关并取全局最大值。
        correlations: list[float] = []
        for tau in range(low, high + 1):
            overlap = frame_len - tau
            # 归一化自相关：分母为两段各自能量，长 lag 不再天然占优。
            num = 0.0
            energy_a = 0.0
            energy_b = 0.0
            for i in range(overlap):
                a = frame[i]
                b = frame[i + tau]
                num += a * b
                energy_a += a * a
                energy_b += b * b
            denom = math.sqrt(energy_a * energy_b)
            correlations.append(
                num / denom if denom > _WINDOW_EPS else 0.0
            )

        max_r = max(correlations)
        # 第二遍：与最大值相差不超过 1e-9 的 lag 中取最小者
        # （周期 80 的帧不会输给 160 的长 lag）。
        best_lag = low
        for offset, r in enumerate(correlations):
            if r >= max_r - 1e-9:
                best_lag = low + offset
                break
        periods.append(int(best_lag))
    return periods


def _stretch_to_length(
    samples: list[float],
    target_len: int,
    frame_len: int,
    hop: int,
) -> list[float]:
    """核心 OLA/WSOLA 拉伸：把 ``samples`` 重做到精确的 ``target_len`` 长。

    与公开 :func:`time_stretch` 的区别是本函数不做 rate 范围校验——变调
    内部需要的等效 rate 可能落在 ``0.1 .. 8.0`` 之外（如 ±48 半音）。
    长度一次性给定，绝不逐帧累加浮点位置。
    """
    n = len(samples)
    if target_len <= 0:
        return []

    # 信号太短、连一帧都放不下时，直接线性插值到目标长度。
    if target_len < frame_len or n < frame_len:
        return _linear_resample(samples, target_len)

    window = _hann_window(frame_len)
    out = [0.0] * target_len
    wsum = [0.0] * target_len

    # 分析位置按合成位置做等比重定位（WSOLA 式，而不是固定 hop）：
    # 目标更长（放慢）时分析位置走得慢，目标更短（加快）时走得快。
    k = 0
    while True:
        syn = k * hop
        if syn >= target_len:
            break
        analysis = int(round(syn * n / target_len))
        # 钳制到源信号内，边界帧取到的仍是真实样本，不补零、不爆音。
        if analysis > n - frame_len:
            analysis = n - frame_len

        for j in range(frame_len):
            pos = syn + j
            if pos >= target_len:
                break
            w = window[j]
            out[pos] += samples[analysis + j] * w
            wsum[pos] += w
        k += 1

    # 除以同位置累计 Hann 窗和，常数信号重建为常数，幅度不被抬高。
    for i in range(target_len):
        if wsum[i] > _WINDOW_EPS:
            out[i] = out[i] / wsum[i]
        else:
            out[i] = 0.0
    return out


def time_stretch(
    samples: list[float],
    rate: float,
    frame_len: int,
    hop: int,
) -> list[float]:
    """WSOLA 式变速不变调：改变时长，保持音高。

    契约
    ----
    * ``samples`` 为单声道样本；空输入返回 ``[]``。
    * ``rate`` 必须为有限正数（``0.1 .. 8.0``），由
      :func:`validate_params` 校验；``rate > 1`` 加快，``rate < 1`` 放慢。
    * ``frame_len`` / ``hop`` 必须满足 ``frame_len > hop >= 1``，否则抛
      :class:`StretchError`。
    * 输出长度一次性计算为 ``round(len(samples) / rate)``，误差不超过
      1 个采样；不得逐帧累加浮点位置，长音频不允许累计 ``10 ms`` 漂移。
      ``rate == 1`` 时长度精确等于输入长度。
    * 使用 Hann 分帧与重叠相加，并按实际窗和归一化，保持幅度有界；
      采样率由调用方在生成样本时决定，本函数不需要额外采样率参数。
      浮点比较容差 ``1e-6``。
    """
    _validate_samples(samples)
    validate_params(rate, 0.0, frame_len)
    _validate_frame_hop(frame_len, hop)

    n = len(samples)
    if n == 0:
        return []

    # 缺陷 3/10：输出长度一次性按 rate 精确计算，绝不逐帧浮点累加。
    target_len = int(round(n / rate))
    return _stretch_to_length(samples, target_len, frame_len, hop)


def pitch_shift(
    samples: list[float],
    semitones: float,
    sample_rate: int,
) -> list[float]:
    """变调不变速：按半音数升降音高，同时保持时长。

    契约
    ----
    * ``sample_rate`` 为采样率（整数 Hz，例如 ``16000``），必须为正。
    * ``semitones`` 为有限浮点数，范围 ``-48 .. 48``；``0`` 时返回输入
      副本，长度与波形在 ``1e-6`` 内完全一致。
    * 输出时长（采样点数）与输入相同，误差不超过 1 个采样；不能只改变
      播放速度而不改变音高。
    * 内部使用约 64 ms 的 Hann 帧（16 kHz 下为 ``frame_len=1024``、
      ``hop=256``，其他采样率按比例调整），重叠相加并归一化窗和；
      空输入返回 ``[]``。
    * 输出样本保持有限并位于 ``-1.0 .. 1.0``；浮点比较容差 ``1e-6``。
    """
    _validate_samples(samples)
    _validate_sample_rate(sample_rate)
    if isinstance(semitones, bool) or not isinstance(semitones, (int, float)):
        raise StretchError("semitones 必须是有限实数")
    semitones = float(semitones)
    if not math.isfinite(semitones):
        raise StretchError("semitones 必须有限")
    if abs(semitones) > _MAX_SEMITONES + TOLERANCE:
        raise StretchError(f"semitones 必须在 -{_MAX_SEMITONES}..{_MAX_SEMITONES}")

    n = len(samples)
    if n == 0:
        return []
    if abs(semitones) <= TOLERANCE:
        # 0 半音：原样副本，波形逐点一致。
        return list(samples)

    # 音高倍率 factor：升调 factor > 1。
    factor = 2.0 ** (semitones / 12.0)

    frame_len, hop = _frame_params_for_rate(sample_rate)
    # 先做变速不变调拉伸到 N*factor 个采样（音高暂未变），再以 factor
    # 重采样回 N 个采样：时长还原、音高实际移动 factor 倍。
    # 注意：极端半音数下等效 rate 可低于 0.1，故直接走内部 OLA，
    # 不经过带 rate 范围校验的公开 time_stretch。
    stretched_len = int(round(n * factor))
    stretched = _stretch_to_length(samples, stretched_len, frame_len, hop)
    out = _linear_resample(stretched, n)

    # 钳制到标称范围，保证有限且有界（窗和归一化后理论上不会越界）。
    return [min(1.0, max(-1.0, v)) for v in out]


def overlap_add(
    frames: list[list[float]],
    hop: int,
) -> list[float]:
    """重叠相加已加窗的帧，并按累计 Hann 窗和归一化。

    契约
    ----
    * ``frames`` 为等长帧（通常来自 :func:`frame_signal`）；空输入返回
      ``[]``。
    * ``frame_len`` 由 ``len(frames[0])`` 推断；``hop`` 为 ``>= 1`` 的整数
      且 ``hop < frame_len``，否则抛 :class:`StretchError`。
    * 输出长度为 ``(len(frames) - 1) * hop + frame_len``。
    * 每个输出采样先累加所有覆盖它的帧采样，再除以同位置累计的 Hann 窗值
      （``w[n] = 0.5 - 0.5 * cos(2*pi*n/(frame_len-1))``）；窗和接近 0 的
      边界位置输出 ``0.0``。因此常数 1.0 的输入在重叠区重建为 1.0，
      而不是 1.5 倍。
    * 不修改输入帧；采样率不参与本函数（帧已加窗），浮点比较容差
      ``1e-6``。
    """
    if not isinstance(frames, list):
        raise StretchError("frames 必须是 list")
    if not frames:
        return []
    if not _is_real_int(hop) or hop < 1:
        raise StretchError("hop 必须是 >= 1 的整数")

    frame_len = len(frames[0])
    if frame_len == 0:
        raise StretchError("帧长度不能为 0")
    if hop >= frame_len:
        raise StretchError("hop 必须小于 frame_len，否则无法重叠相加")
    for frame in frames:
        if len(frame) != frame_len:
            raise StretchError("所有帧必须等长")

    window = _hann_window(frame_len)
    out_len = (len(frames) - 1) * hop + frame_len
    acc = [0.0] * out_len
    wsum = [0.0] * out_len

    for k, frame in enumerate(frames):
        start = k * hop
        for j in range(frame_len):
            pos = start + j
            acc[pos] += frame[j]
            wsum[pos] += window[j]

    # 除以窗和：重叠区常数信号精确重建为 1.0；边界零区输出 0.0。
    return [
        acc[i] / wsum[i] if wsum[i] > _WINDOW_EPS else 0.0
        for i in range(out_len)
    ]


def preserve_formant(
    samples: list[float],
    semitones: float,
    sample_rate: int,
    strength: float,
) -> list[float]:
    """变调时补偿共振峰，避免“花栗鼠”或过分低沉的失真。

    契约
    ----
    * ``sample_rate`` 为正整数 Hz；``semitones`` 为有限浮点数
      （``-48 .. 48``）。
    * ``strength`` 表示共振峰补偿强度：``0`` 等价于普通
      :func:`pitch_shift`，``1`` 为完全补偿；传入小于 0 或大于 1 的值时
      **钳制** 到 ``0 .. 1``，不得因 ``1.5``、``2.0`` 之类输入产生失真。
      非有限 ``strength`` 抛 :class:`StretchError`。
    * 输出长度与输入相同（误差不超过 1 个采样）；空输入返回 ``[]``。
    * 采样率参与内部帧长、hop 与重采样换算（16 kHz 下约为 1024/256）；
      浮点比较容差 ``1e-6``。
    """
    _validate_samples(samples)
    _validate_sample_rate(sample_rate)
    if isinstance(strength, bool) or not isinstance(strength, (int, float)):
        raise StretchError("strength 必须是有限实数")
    strength = float(strength)
    if not math.isfinite(strength):
        raise StretchError("strength 必须有限，NaN/inf 不允许")
    # 缺陷 7：超界强度钳制到 [0, 1]，而不是放大到 >1 导致失真。
    strength = min(1.0, max(0.0, strength))

    n = len(samples)
    if n == 0:
        return []

    shifted = pitch_shift(samples, semitones, sample_rate)
    if strength <= 0.0:
        return shifted

    # 共振峰补偿：用源信号与变调信号的短时幅度包络之比，把变调结果的
    # 能量包络拉回原信号。包络恒正且补偿量受 strength 钳制，不会超过 1。
    frame_len, _ = _frame_params_for_rate(sample_rate)
    env_source = _moving_average_envelope(samples, frame_len)
    env_shifted = _moving_average_envelope(shifted, frame_len)

    floor = 1e-8
    out: list[float] = []
    for i in range(n):
        ratio = env_source[i] / max(env_shifted[i], floor)
        # 完全补偿(strength=1)用原始包络比；部分补偿在 1.0 与比值间插值，
        # 保证 strength<=1 时增益不会无界放大。
        gain = 1.0 + strength * (ratio - 1.0)
        value = shifted[i] * gain
        out.append(min(1.0, max(-1.0, value)))
    return out


def validate_params(
    rate: float,
    semitones: float,
    frame_len: int,
) -> None:
    """校验拉伸/变调共用参数，非法时抛 :class:`StretchError`。

    契约
    ----
    * ``rate``：有限实数，满足 ``0.1 <= rate <= 8.0``。
    * ``semitones``：有限实数，满足 ``-48 <= semitones <= 48``。
    * ``frame_len``：整数，满足 ``2 <= frame_len <= 65536``。
    * 任一条件不满足（``NaN``、``inf``、0、负数、非整数帧长等）都抛
      ``StretchError``；合法时返回 ``None``，不做任何 I/O。
    * 采样率与 hop 不在本函数签名内：需要它们的函数
      （:func:`pitch_shift`、:func:`preserve_formant`、
      :func:`time_stretch`、:func:`frame_signal`）必须各自校验
      ``sample_rate > 0`` 与 ``frame_len > hop >= 1``，避免 hop 为 0 时
      除零。浮点边界比较容差 ``1e-6``。
    """
    # rate
    if isinstance(rate, bool) or not isinstance(rate, (int, float)):
        raise StretchError("rate 必须是有限实数")
    rate = float(rate)
    if not math.isfinite(rate):
        raise StretchError("rate 必须有限，NaN/inf 不允许")
    if rate < _MIN_RATE - TOLERANCE or rate > _MAX_RATE + TOLERANCE:
        raise StretchError(f"rate 必须在 {_MIN_RATE}..{_MAX_RATE} 之间")

    # semitones
    if isinstance(semitones, bool) or not isinstance(
        semitones, (int, float)
    ):
        raise StretchError("semitones 必须是有限实数")
    semitones = float(semitones)
    if not math.isfinite(semitones):
        raise StretchError("semitones 必须有限，NaN/inf 不允许")
    if semitones < -_MAX_SEMITONES - TOLERANCE or (
        semitones > _MAX_SEMITONES + TOLERANCE
    ):
        raise StretchError(
            f"semitones 必须在 -{_MAX_SEMITONES}..{_MAX_SEMITONES} 之间"
        )

    # frame_len：必须是真整数（bool 排除）且有界。
    if not _is_real_int(frame_len):
        raise StretchError("frame_len 必须是整数")
    if frame_len < _MIN_FRAME_LEN or frame_len > _MAX_FRAME_LEN:
        raise StretchError(
            f"frame_len 必须在 {_MIN_FRAME_LEN}..{_MAX_FRAME_LEN} 之间"
        )


def process_batch(
    tracks: list[list[float]],
    params: dict,
) -> dict:
    """批量处理多条单声道音轨，任一轨非法则整批回滚。

    契约
    ----
    * ``tracks`` 为 ``list[list[float]]``；空列表返回
      ``{"tracks": [], "count": 0}``。
    * ``params`` 为字典：必填 ``"rate"``、``"semitones"``、
      ``"sample_rate"``（采样率，整数 Hz），可选 ``"frame_len"``
      （帧长 1024）、``"hop"``（默认 256）、``"strength"``
      （默认 0.0）。
    * 每条轨道独立处理：``strength > 0`` 用 :func:`preserve_formant`，
      否则用 :func:`pitch_shift`；当 ``rate != 1`` 再对结果做
      :func:`time_stretch`。每次调用使用全新的分析缓冲与输出列表，
      轨道之间不得串音。
    * 全有或全无：先校验参数与每一条轨道，把成功结果写入局部列表；任一轨
      非法（非序列、非数值、越界、``NaN`` 等）抛 :class:`StretchError`，
      不返回部分结果，实现整体回滚。
    * 全部成功时返回 ``{"tracks": [...], "count": len(tracks)}``，顺序与
      输入一致；浮点比较容差 ``1e-6``。
    """
    if not isinstance(tracks, list):
        raise StretchError("tracks 必须是 list[list[float]]")
    if not isinstance(params, dict):
        raise StretchError("params 必须是 dict")
    if not tracks:
        return {"tracks": [], "count": 0}

    try:
        rate = params["rate"]
        semitones = params["semitones"]
        sample_rate = params["sample_rate"]
    except KeyError as exc:
        raise StretchError(f"params 缺少必填字段: {exc.args[0]}") from exc

    frame_len = params.get("frame_len", DEFAULT_FRAME_LEN)
    hop = params.get("hop", DEFAULT_HOP)
    strength = params.get("strength", 0.0)

    # 先统一校验参数，任何一条轨非法都不会产生部分输出。
    validate_params(rate, semitones, frame_len)
    _validate_sample_rate(sample_rate)
    _validate_frame_hop(frame_len, hop)
    if isinstance(strength, bool) or not isinstance(strength, (int, float)):
        raise StretchError("strength 必须是有限实数")
    if not math.isfinite(float(strength)):
        raise StretchError("strength 必须有限")

    # 先校验全部轨道（全有或全无），再逐条处理。
    for track in tracks:
        _validate_samples(track)

    results: list[list[float]] = []
    for track in tracks:
        # 每条轨都拿到全新的局部缓冲，函数间不共享任何可变状态，杜绝串音。
        if strength > 0.0:
            processed = preserve_formant(
                track, semitones, sample_rate, strength
            )
        else:
            processed = pitch_shift(track, semitones, sample_rate)

        if abs(float(rate) - 1.0) > TOLERANCE:
            processed = time_stretch(processed, rate, frame_len, hop)
        results.append(processed)

    return {"tracks": results, "count": len(tracks)}

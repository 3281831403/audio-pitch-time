"""15 个红灯测试：每个测试都直接调用尚未实现的桩函数。

测试只验证公开契约，不修改实现。浮点比较统一使用 1e-6 绝对容差。
"""

import math

import pytest

from pitchshift.stretch import (
    StretchError,
    detect_period,
    frame_signal,
    overlap_add,
    pitch_shift,
    preserve_formant,
    process_batch,
    time_stretch,
    validate_params,
)


SAMPLE_RATE = 16000


def _tone(freq=220.0, seconds=0.5, sample_rate=SAMPLE_RATE, amp=0.5):
    """生成确定性的正弦测试信号，不使用随机数。"""
    n = int(round(seconds * sample_rate))
    return [amp * math.sin(2.0 * math.pi * freq * i / sample_rate) for i in range(n)]


def _zero_crossing_freq(samples, sample_rate):
    """用中间 80% 波形的过零率估计纯音频率。"""
    margin = len(samples) // 10
    segment = samples[margin : len(samples) - margin]
    if len(segment) < 2:
        return 0.0
    crossings = sum(
        1
        for a, b in zip(segment, segment[1:])
        if (a <= 0.0 < b) or (b <= 0.0 < a)
    )
    return crossings / (2.0 * len(segment) / sample_rate)


def test_frame_signal_applies_hann_window():
    """缺陷 1：分帧必须加 Hann 窗，否则重叠相加出现梳状滤波。"""
    frames = frame_signal([1.0] * 65, frame_len=33, hop=16)

    assert len(frames) == 3
    assert all(len(frame) == 33 for frame in frames)

    first = frames[0]
    assert first[0] == pytest.approx(0.0, abs=1e-9)
    assert first[-1] == pytest.approx(0.0, abs=1e-9)
    assert first[16] == pytest.approx(1.0, abs=1e-6)
    assert first[8] == pytest.approx(0.5, abs=1e-6)
    assert first[4] == pytest.approx(0.5 - 0.5 * math.cos(math.pi / 4.0), abs=1e-6)
    # 未加窗时所有采样都会保持 1.0。
    assert any(abs(value - 1.0) > 1e-3 for value in first)


def test_detect_period_prefers_normalized_peak():
    """缺陷 2：归一化自相关应选真实周期 80，而不是更长的 160。"""
    period = 80
    frame = [math.sin(2.0 * math.pi * i / period) for i in range(1024)]
    shifted = [math.sin(2.0 * math.pi * (i + 20) / period) for i in range(1024)]

    assert detect_period([frame, shifted], min_lag=40, max_lag=160) == [80, 80]

    clamped = detect_period([frame], min_lag=-10, max_lag=10**6)
    assert len(clamped) == 1
    assert isinstance(clamped[0], int)
    assert 2 <= clamped[0] <= 1023


def test_time_stretch_respects_rate_ratio():
    """缺陷 3：输出长度必须随 rate 变化，不能固定 hop。"""
    samples = _tone(220.0, 0.5)

    for rate in (0.5, 1.5, 2.0):
        out = time_stretch(samples, rate, frame_len=512, hop=128)
        assert abs(len(out) - len(samples) / rate) <= 1.0


def test_pitch_shift_changes_pitch_keeps_duration():
    """缺陷 4：升 12 个半音应让频率翻倍，同时时长不变。"""
    samples = _tone(400.0, 0.5)

    out = pitch_shift(samples, semitones=12.0, sample_rate=SAMPLE_RATE)

    assert abs(len(out) - len(samples)) <= 1
    base_freq = _zero_crossing_freq(samples, SAMPLE_RATE)
    shifted_freq = _zero_crossing_freq(out, SAMPLE_RATE)
    assert shifted_freq == pytest.approx(2.0 * base_freq, rel=0.15)


def test_overlap_add_normalizes_window_sum():
    """缺陷 5：重叠相加必须除以窗和，不能把音量抬高 1.5 倍。"""
    frames = frame_signal([1.0] * 1024, frame_len=64, hop=16)

    out = overlap_add(frames, hop=16)

    assert len(out) == 1024
    interior = out[64:960]
    assert max(abs(value - 1.0) for value in interior) <= 1e-6


def test_boundaries_do_not_click():
    """缺陷 6：边界不能因补零出现大幅跳变（爆音）。"""
    samples = _tone(440.0, 0.2)

    out = pitch_shift(samples, semitones=5.0, sample_rate=SAMPLE_RATE)

    assert len(out) > 1
    steps = [abs(out[i + 1] - out[i]) for i in range(len(out) - 1)]
    assert max(steps) <= 0.25


def test_preserve_formant_clamps_strength():
    """缺陷 7：strength 超界必须钳制，不能产生失真。"""
    samples = _tone(220.0, 0.2)

    full = preserve_formant(samples, 7.0, SAMPLE_RATE, strength=1.0)
    over = preserve_formant(samples, 7.0, SAMPLE_RATE, strength=2.5)
    assert len(over) == len(full)
    assert max(abs(a - b) for a, b in zip(over, full)) <= 1e-6

    plain = preserve_formant(samples, 7.0, SAMPLE_RATE, strength=0.0)
    under = preserve_formant(samples, 7.0, SAMPLE_RATE, strength=-3.0)
    assert max(abs(a - b) for a, b in zip(under, plain)) <= 1e-6


def test_validate_params_rejects_illegal_values():
    """缺陷 8：非法参数必须抛 StretchError，不能继续执行到除零。"""
    assert validate_params(1.0, 0.0, 1024) is None

    for rate, semitones, frame_len in (
        (0.0, 0.0, 1024),
        (-1.0, 0.0, 1024),
        (float("nan"), 0.0, 1024),
        (1.0, 1000.0, 1024),
        (1.0, 0.0, 0),
    ):
        with pytest.raises(StretchError):
            validate_params(rate, semitones, frame_len)


def test_process_batch_isolates_tracks():
    """缺陷 9：批量处理不能共用分析缓冲，静音轨不能被串音污染。"""
    params = {
        "rate": 1.0,
        "semitones": 7.0,
        "sample_rate": SAMPLE_RATE,
        "frame_len": 512,
        "hop": 128,
    }
    voice = _tone(220.0, 0.2)
    silence = [0.0] * len(voice)

    batch = process_batch([voice, silence], params)
    solo = process_batch([voice], params)

    assert batch["count"] == 2
    assert len(batch["tracks"]) == 2
    assert max(
        abs(a - b) for a, b in zip(batch["tracks"][0], solo["tracks"][0])
    ) <= 1e-6
    assert max(abs(value) for value in batch["tracks"][1]) <= 1e-9


def test_long_signal_has_no_length_drift():
    """缺陷 10：长音频长度用一次性取整，不能累计漂移 10 ms。"""
    samples = _tone(220.0, 10.0)

    for rate in (1.3, 0.7):
        out = time_stretch(samples, rate, frame_len=1024, hop=256)
        assert abs(len(out) - len(samples) / rate) <= 1.0


def test_rate_one_preserves_length():
    """组合：rate=1 时输出长度必须等于输入长度。"""
    samples = _tone(330.0, 0.5)

    assert len(time_stretch(samples, 1.0, frame_len=512, hop=128)) == len(samples)


def test_zero_semitones_preserves_waveform():
    """组合：semitones=0 时波形不变（绝对容差 1e-6）。"""
    samples = _tone(330.0, 0.2)

    out = pitch_shift(samples, 0.0, SAMPLE_RATE)

    assert len(out) == len(samples)
    assert max(abs(a - b) for a, b in zip(out, samples)) <= 1e-6


def test_empty_input_returns_empty():
    """组合：所有入口的空输入都返回空结果。"""
    assert frame_signal([], 1024, 256) == []
    assert detect_period([], 20, 200) == []
    assert time_stretch([], 1.5, 1024, 256) == []
    assert pitch_shift([], 12.0, SAMPLE_RATE) == []
    assert overlap_add([], 256) == []
    assert preserve_formant([], 12.0, SAMPLE_RATE, 0.5) == []
    assert process_batch(
        [],
        {"rate": 1.0, "semitones": 0.0, "sample_rate": SAMPLE_RATE},
    ) == {"tracks": [], "count": 0}


def test_frame_len_not_greater_than_hop_raises():
    """组合：frame_len <= hop 必须抛 StretchError，不能除零。"""
    samples = _tone(220.0, 0.1)

    with pytest.raises(StretchError):
        time_stretch(samples, 1.0, frame_len=256, hop=256)

    with pytest.raises(StretchError):
        frame_signal(samples, frame_len=128, hop=256)


def test_process_batch_preserves_order_on_success():
    """组合：批量全部成功时输出顺序与输入一致。"""
    tones = [_tone(freq, 0.2) for freq in (220.0, 330.0, 440.0)]
    params = {
        "rate": 1.0,
        "semitones": 12.0,
        "sample_rate": SAMPLE_RATE,
        "frame_len": 512,
        "hop": 128,
    }

    result = process_batch(tones, params)

    assert result["count"] == 3
    assert len(result["tracks"]) == 3
    for actual, source in zip(result["tracks"], tones):
        expected = pitch_shift(source, 12.0, SAMPLE_RATE)
        assert abs(len(actual) - len(expected)) <= 1
        assert max(abs(a - b) for a, b in zip(actual, expected)) <= 1e-6

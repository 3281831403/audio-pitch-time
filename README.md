# audio-pitch-time

## 题面

播客做变速不变调后，人声发闷还带金属味：分帧不加 Hann 窗，重叠相加出现梳状滤波；自相关不归一化，长 lag 永远胜出；拉伸按固定 hop，rate 变化时长度比例错；变调只改速度不改音高，听起来只是变速；重叠相加不除以窗和，整体音量涨 1.5 倍；边界帧补零，首尾出现爆音；共振峰强度不钳制，超过 1 后失真；参数校验不查 frame_len 是否大于 hop，除零；批量共用分析缓冲，轨道间串音；输出长度用浮点四舍五入，长音频累计漂移 10 ms。pitchshift/stretch.py 是核心，tests/test_stretch.py 15个红。要修的是：把上述 10 个缺陷逐一修掉，让 15 个测试全部转绿，同时保持 `list[float]` 接口、长度按采样精确计算、异常统一为 `StretchError`。

## 运行方式

在仓库根目录执行：

```bash
python -m pytest tests -q
```

初始（未实现）状态下应恰好输出 `15 failed`，且没有 collection error、import error 或 syntax error：

```text
15 failed
```

每个测试都直接调用 `pitchshift/stretch.py` 中的桩函数，因此当前失败原因是 `NotImplementedError`。实现完成后应变为 `15 passed`。

## 字段表

| 字段 | 取值 |
| --- | --- |
| 所属领域 | 游戏、图形与多媒体 |
| 子方向 | 音视频处理 |
| 任务类型 | Bugfix |
| 语言/框架 | Python 3, pytest |
| 环境可复现等级 | 无外部依赖 |

## 目录结构

```text
audio-pitch-time/
├── README.md
├── conftest.py
├── .gitignore
├── pitchshift/
│   ├── __init__.py
│   └── stretch.py
└── tests/
    └── test_stretch.py
```

`conftest.py` 将仓库根加入 `sys.path`，因此测试无需安装包即可导入 `pitchshift`。实现只使用标准库 `math`（必要时可用 `cmath`），不联网、不使用随机数、不调用 `sleep`。

## 接口

`pitchshift/stretch.py` 提供 8 个待实现函数：

| 函数 | 作用 |
| --- | --- |
| `frame_signal(samples, frame_len, hop)` | 分帧并加 Hann 窗，只输出完整帧，不补零 |
| `detect_period(frames, min_lag, max_lag)` | 归一化自相关估计基频周期，越界钳制 |
| `time_stretch(samples, rate, frame_len, hop)` | WSOLA 式变速不变调，长度误差不超过 1 采样 |
| `pitch_shift(samples, semitones, sample_rate)` | 变调不变速，长度误差不超过 1 采样 |
| `overlap_add(frames, hop)` | 重叠相加并除以累计窗和 |
| `preserve_formant(samples, semitones, sample_rate, strength)` | 共振峰补偿，`strength` 钳制到 `0..1` |
| `validate_params(rate, semitones, frame_len)` | 校验参数，非法时抛 `StretchError` |
| `process_batch(tracks, params)` | 批量处理，任一轨非法则整批回滚、不串音 |

## 约束

- 样本统一为 `list[float]`，标称范围 `-1.0 .. 1.0`；所有长度、帧长与 hop 均以采样点为单位，采样率为整数 Hz（例如 `16000`）。
- 分帧与拉伸要求 `frame_len > hop >= 1`；浮点比较容差为 `1e-6`。
- 异常统一使用 `pitchshift.stretch.StretchError`。
- `tests/test_stretch.py` 不可修改；只允许修改 `pitchshift/stretch.py`（必要时可调整 `pitchshift/__init__.py` 的导出）。

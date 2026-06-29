# CARRY Beamforming Parameter Generation

## 1. 这个项目是做什么的

这个仓库的主要用途是为 **CARRY 32 波束 trace 跟踪模式** 生成和预览参数文件。

当前主流程围绕 `compute_trace_mode_phase_coeff.py` 展开。它会根据：

- 天线阵列文件 `ants.txt`
- 目标源赤经/赤纬 `TARGET_RA` / `TARGET_DEC`
- 最小仰角 `MIN_ELEVATION_DEG`
- 外部 32 波束偏移表 `config_32beam_hex37_drop5_beam_offsets.txt`

连续计算每个 10 秒时隙（slot）对应的：

- `A[i, j]`
- `B[i, j]`
- `omega[i, j]`

然后把它们按固定硬件契约写成一个二进制浮点流文件：

- `trace_mode_phase_coeff.txt`

虽然文件名后缀是 `.txt`，但它的实际内容是 **little-endian float32 二进制数据**，不是纯文本。

这个仓库还提供：

- 一个 GUI 预览工具 `trace_32beam_phase_coeff_gui.py`
- 若干辅助脚本，用于查看、编辑或生成其他相关参数文件


## 2. 如果你只想快速知道主入口

最重要的文件只有三个：

1. `compute_trace_mode_phase_coeff.py`
   - 当前主程序
   - 负责持续运行、按 10 秒时隙计算并发布 `trace_mode_phase_coeff.txt`

2. `beam32_from_azel_functions.py`
   - 核心公共函数库
   - 负责时间、坐标、天线、波束方向、几何基底等底层计算

3. `trace_32beam_phase_coeff_gui.py`
   - GUI 预览工具
   - 不负责真正发布参数文件
   - 只负责预览“当前状态 / 下一 slot / 32 波束方向 / 输出文件状态”


## 3. 这个项目最终输出什么

### 主输出文件

- `trace_mode_phase_coeff.txt`

它的输出契约是固定的：

- 逻辑布局：`A[20,32] + B[20,32] + omega[20,32]`
- 数据类型：`little-endian float32`
- 总值数量：`3 * 20 * 32 = 1920`
- 总字节数：`1920 * 4 = 7680 bytes`

其中：

- 行数固定为 `20`，代表固定硬件输入总数 `TOTAL_SIGNAL_INPUTS = 20`
- 列数固定为 `32`，代表固定波束数 `TOTAL_BEAMS = 32`

注意：

- `A/B` **不包含频率项**
- 频率 `nu` 由 GPU 端代入
- `t0` **不写入参数文件**

主程序注释里给出的 GPU 侧公式是：

```text
Phi3 = nu * (A*cos(delta) + B*sin(delta) - A)
delta = omega * (t - t0)
```

这里 `t0` 只在使用端的时序里出现，不作为一个字段写进 `trace_mode_phase_coeff.txt`。


## 4. 主程序到底在做什么

`compute_trace_mode_phase_coeff.py` 不再是“一次性算完就退出”的脚本，而是一个 **持续运行的 10 秒时隙发布器**。

它的整体思路是：

1. 建立一个与 UTC 10 秒网格对齐的连续跟踪会话
2. 预计算未来若干个 slot 的参数包
3. 在每个 slot 到来前一点点时间发布输出文件
4. 原子替换 `trace_mode_phase_coeff.txt`

也就是说，这个程序的目标不是“给你一份离线报告”，而是“持续给后端/硬件提供下一时刻要用的波束参数”。


## 5. 主代码流程，按执行顺序说明

下面这部分是理解整个项目最关键的内容。

### Step 1: 读取用户配置

主程序开头定义了一组用户可调参数：

- `ANTS_TXT`
- `TARGET_RA`
- `TARGET_DEC`
- `MIN_ELEVATION_DEG`
- `SIMULATION_IGNORE_VISIBILITY`
- `CENTER_FREQ_HZ`
- `OMEGA_DELTA_SECONDS`
- `UPDATE_PERIOD_SECONDS`
- `PUBLISH_LEAD_SECONDS`
- `PRECOMPUTE_QUEUE_SLOTS`
- `BEAM_OFFSET_TABLE_FILE`

其中最关键的几个是：

- `SIMULATION_IGNORE_VISIBILITY`
  - 如果目标源当前不可见：
    - `False`：正常观测逻辑，不发布
    - `True`：仍然继续计算并发布，作为模拟/联调用途

- `BEAM_OFFSET_TABLE_FILE`
  - 当前版本要求 **32 波束偏移必须来自外部文件**
  - 自动生成 32 波束布局的逻辑已经禁用


### Step 2: 建立 session，并对齐到 10 秒边界

主程序通过 `build_session_context()` 建立一个 `SessionContext`：

- 记录程序启动时间
- 生成本次 session 的 `session_id`
- 计算 `t_ref_utc`

这个 `t_ref_utc` 不是任意时刻，而是通过：

- `ceil_utc_to_next_slot_boundary(...)`

向上对齐到下一个 10 秒边界。例如：

- 当前是 `12:00:03`
- 下一个 slot 边界可能就是 `12:00:10`

这样整个发布过程就和固定 10 秒时隙网格绑定起来了。


### Step 3: 读取天线文件 `ants.txt`

主程序通过 `read_antenna_file(...)` 读取天线配置。

`ants.txt` 每一行格式是：

```text
name lat lon height diameter
```

仓库里的示例是：

```text
# name lat lon [alt_m] [diam_m]
ant20 29.784402 109.779625 1581 7.5
ant17 29.78445918504143 109.78025550000001 1588 7.5
ant21 29.783878 109.779485 1593 7.5
ant24 29.784690 109.779090 1593 7.5
```

读取后，程序会把每面天线转换成：

- 地理坐标
- ECEF 坐标
- 相对参考天线的 ENU 坐标

这些信息被封装到 `AntennaRecord` 中。

### 关于输入路数

当前硬件契约固定为：

- 每面天线对应 `2` 路信号
- 总硬件输入固定为 `20`

也就是说：

- `active_signal_inputs = antenna_count * 2`
- 如果少于 20 路，其余输入全部补 `0`

例如 4 面天线时：

- active inputs = `8`
- padded inputs = `12`


### Step 4: 读取并校验 32 波束偏移表

主程序通过 `resolve_beam_offset_table(...)` 读取：

- `config_32beam_hex37_drop5_beam_offsets.txt`

当前版本的规则非常明确：

- 必须来自外部文件
- 必须正好有 `32` 行波束
- 外部 `BeamID` 可以是：
  - `1..32`
  - 或 `0..31`
- 内部统一转换成：
  - `beam_index = 0..31`

文件里真正会被读取的核心列至少包括：

```text
BeamID  q  r  dEast_deg  dNorth_deg
```

仓库中的示例文件除了这些列，还额外附带了：

- `offset_deg`
- `PA_deg`
- 说明文字

这些扩展内容是允许存在的，程序会跳过表头说明，提取有效数据行。

### 为什么这一步很重要

因为当前程序已经不再自己“推导一套 32 波束排布”，而是把“波束排布定义”视为一个外部输入契约。

也就是说：

- 机械指向中心由 `TARGET_RA/TARGET_DEC` 决定
- 32 个 beam 相对中心的偏移由外部文件决定

这让波束布局可控、可审计，也更符合工程联调场景。


### Step 5: 计算目标源当前状态和可见性

主程序会通过：

- `build_live_status_snapshot(...)`
- `compute_visibility_window(...)`

计算目标源在某个时刻的：

- 当前方位角 `az`
- 当前仰角 `el`
- 本地恒星时 `lst`
- 当前是否可见
- 当前可见窗口开始/结束
- 下次升起/下次过中天等信息

这一步既用于：

- 启动时打印状态
- GUI 预览
- 每个 slot 判断是否应该发布


### Step 6: 为某个 slot 构建 32 波束方向

每个 slot 的核心计算从 `compute_beam_model_state(...)` 开始。

它会做两件事：

1. 计算主波束（beam0）在当前时刻的目标源方向
2. 基于 32 波束偏移表，为每根天线计算 32 个 beam 的方向向量和 az/el

这里底层调用的是：

- `trace_collect_antenna_beam_results(...)`

而这个函数来自 `beam32_from_azel_functions.py`。

换句话说，`beam32_from_azel_functions.py` 负责的是：

- 天球坐标
- 水平坐标
- ENU 几何基底
- 波束方向向量

这些“几何和时间基础设施”。


### Step 7: 计算 A / B / omega

这是整个项目的核心数值计算部分，对应：

- `compute_abomega(...)`

它会针对每个天线、每个 beam，计算：

- `A[N_ant, 32]`
- `B[N_ant, 32]`
- `omega[N_ant, 32]`

其做法大致是：

1. 在 `t - OMEGA_DELTA_SECONDS`、`t`、`t + OMEGA_DELTA_SECONDS` 三个时刻计算 beam 方向
2. 用前后相位差估计角速度 `omega`
3. 在源对齐的局部水平参考系下，把天线基线投影到 `x/y` 方向
4. 得到每个 beam 的 `A` 和 `B`

然后再通过：

- `expand_antennas_to_twenty_rows(...)`

把 `N_ant x 32` 扩展成固定的 `20 x 32`：

- 每面天线复制成 2 路输入
- 不足的输入行全部补零


### Step 8: 组装成硬件文件格式

发布之前，主程序会把三块矩阵直接拼接：

```text
A.reshape(-1) + B.reshape(-1) + omega.reshape(-1)
```

也就是：

1. 先写完整的 `A[20,32]`
2. 再写完整的 `B[20,32]`
3. 最后写完整的 `omega[20,32]`

最终总数必须严格等于：

- `TRACE_STREAM_FLOAT_COUNT = 1920`

否则程序会报错，不会写出错误文件。


### Step 9: 原子发布输出文件

真正写文件时使用的是：

- `write_float32_le_temp_file(...)`
- `atomic_replace(...)`

也就是说：

1. 先写一个临时文件
2. `flush + fsync`
3. 再原子替换正式文件

这么做的目的是避免下游在读文件时撞上“写了一半的文件”。


### Step 10: 多线程连续发布

主程序的持续运行由两个线程协作完成：

1. `precompute_worker`
   - 负责提前计算未来 slot 的参数包

2. `publisher_worker`
   - 负责按时序把参数包真正发布出去

中间通过一个有界队列传递 `SlotPackage`。

这样设计的好处是：

- 计算和发布解耦
- 发布时刻更稳定
- 未来若干 slot 可以提前准备


## 6. GUI 是干什么的

`trace_32beam_phase_coeff_gui.py` 是一个 **预览工具**，不是连续发布器。

它的职责是：

1. 显示目标源当前状态
2. 显示下一 10 秒 slot 状态
3. 显示望远镜示意图
4. 显示 32 波束排布
5. 显示当前每个 beam 的 az/el
6. 显示输入路数映射
7. 显示外部 32 波束文件状态
8. 显示 `trace_mode_phase_coeff.txt` 文件状态

它内部会复用主程序里的：

- `build_live_status_snapshot(...)`
- `resolve_beam_offset_table(...)`
- `trace_collect_antenna_beam_results(...)`

所以 GUI 和主程序的逻辑是对齐的。

### GUI 不做什么

GUI 默认不负责长期阻塞运行的连续发布。

不过它提供了一个按钮，可以通过子进程启动：

- `compute_trace_mode_phase_coeff.py`


## 7. 主要文件说明

### 主链路文件

- `compute_trace_mode_phase_coeff.py`
  - 当前主程序
  - 负责连续 slot 计算与发布

- `beam32_from_azel_functions.py`
  - 公共数学/几何/坐标/波束计算函数

- `trace_32beam_phase_coeff_gui.py`
  - 当前主 GUI 预览器

- `ants.txt`
  - 天线位置与口径输入

- `config_32beam_hex37_drop5_beam_offsets.txt`
  - 外部 32 波束偏移定义

### 运行后会生成的主要文件

- `trace_mode_phase_coeff.txt`
  - 主输出

- `trace_32beam_beam_offsets.txt`
  - 当前使用的 32 波束偏移表报告

- `trace_32beam_beam_directions.txt`
  - 当前时刻每个 beam 的方向报告

- `trace_32beam_phase_coeff.log`
  - JSON 行格式日志

### 辅助 / 非主链路脚本

- `compute_beamforming_reference_delay_from_txt.py`
  - 根据 `cable_relative_delay_ns.txt` 生成时间延迟补偿参数
  - 这是另一条辅助流程，不是当前 trace 发布主链路

- `beam_coeff_antenna_editor.py`
  - 用 GUI 编辑 `20 x 32` 的 beam enable 矩阵

- `compute_beam_coeff_external_tmp.py`
  - 一个相对独立的波束系数/方向向量计算脚本
  - 更适合离线检查和诊断，不是当前主发布器


## 8. 关键输入文件格式

### 8.1 `ants.txt`

格式：

```text
name lat lon height diameter
```

示例：

```text
ant20 29.784402 109.779625 1581 7.5
ant17 29.78445918504143 109.78025550000001 1588 7.5
```

含义：

- `name`：天线名
- `lat`：纬度（度）
- `lon`：经度（度）
- `height`：海拔/高度（米）
- `diameter`：口径（米）


### 8.2 `config_32beam_hex37_drop5_beam_offsets.txt`

程序至少需要读取以下列：

```text
BeamID q r dEast_deg dNorth_deg
```

含义：

- `BeamID`
  - 外部 beam 标识
  - 可以是 `1..32` 或 `0..31`

- `q, r`
  - 六边形布局时的轴坐标
  - 如果布局不是 hex，也可以作为辅助列

- `dEast_deg`
  - 相对机械指向中心的东向偏移（度）

- `dNorth_deg`
  - 相对机械指向中心的北向偏移（度）

程序会把它们统一转换为内部：

- `beam_index = 0..31`


## 9. 运行方式

### 9.1 运行连续发布主程序

```bash
python compute_trace_mode_phase_coeff.py
```

运行后会：

- 建立 session
- 读取天线
- 校验外部 32 波束偏移表
- 持续按 10 秒 slot 计算
- 不断更新 `trace_mode_phase_coeff.txt`


### 9.2 打开 GUI 预览器

```bash
python trace_32beam_phase_coeff_gui.py
```

GUI 会显示：

- `Status` 页
  - Current source status
  - Next slot preview
  - Telescope view

- `32-beam layout` 页
  - 32 波束大图
  - beam detail

- `Output` 页
  - input mapping
  - beam offset file
  - trace output file

- `Log` 页
  - 刷新日志和 traceback


### 9.3 运行延迟补偿辅助脚本

```bash
python compute_beamforming_reference_delay_from_txt.py
```

这个脚本不是 trace 主流程的一部分，它处理的是：

- 电缆相对延迟
- 时间延迟量化
- `time_phase_coeff.dat / .npz`


## 10. 依赖环境

主链路代码依赖的第三方库主要包括：

- `numpy`
- `ephem`
- `katpoint`

GUI 依赖：

- `tkinter`

在常见 Windows Python 环境里，`tkinter` 通常随 CPython 自带。


## 11. 读代码时建议从哪里开始

如果你是第一次接手这个项目，建议按下面顺序看：

1. 先看本 README
2. 再看 `compute_trace_mode_phase_coeff.py` 顶部配置区
3. 看 `run_trace_phase_coeff_session()`
4. 看 `build_slot_package()`
5. 看 `compute_abomega()`
6. 遇到几何/坐标细节时再跳到 `beam32_from_azel_functions.py`
7. 想验证运行状态时再打开 `trace_32beam_phase_coeff_gui.py`

这样最容易先建立“工程流程”的整体理解，再进入公式和几何细节。


## 12. 一句话总结

这个仓库的核心不是“画 32 波束图”，也不是“离线生成一份报告”，而是：

**在给定天线阵列、目标源和外部 32 波束偏移表的前提下，持续为 trace 跟踪模式生成符合硬件契约的 `A/B/omega` 参数文件 `trace_mode_phase_coeff.txt`。**

# FineVideo 音频 Caption 数据集构建方案（Qwen3-Omni 对齐版）

> 基于 [HuggingFaceFV/finevideo](https://huggingface.co/datasets/HuggingFaceFV/finevideo) 构建音频 caption / 时序理解测试数据集。  
> 输入为 **独立 WAV 音频 + 5–6 张均匀抽帧图像**（不传整段 mp4），**问题一律英文**，输出格式对齐 Qwen3-Omni；verl-omni rollout 使用 **vLLM-Omni** backend。

---

## 1. 任务理解

| 维度 | 要求 |
|------|------|
| 原始标注 | JSON，含 title + 起止时间戳 + description |
| 模型输入 | **独立 WAV 音频** + **5–6 张均匀抽帧图像**（不是 mp4） |
| 任务形式 | 给定问题，从 description 中选/生成正确答案 |
| 正负样本 | 可自由构造（同视频其他片段、其他视频、模板化干扰项均可） |
| **问题语言** | parquet 中所有 **question / instruction 必须为英文**（FineVideo 标注本身为英文） |
| **Rollout backend** | Qwen3-Omni 训练/评测 rollout 使用 **vLLM-Omni**（`rollout.name=vllm_omni`） |

FineVideo 天然具备这些字段，但需要 **从「整段视频 + 嵌套 JSON」改造成「片段级 multimodal 样本」**。

---

## 2. FineVideo 可用字段映射

FineVideo 每条样本 = `mp4` + `json`，其中与时间片段 caption 最相关的是：

```
FineVideo 单条样本
├── mp4（整段视频）
└── json 元数据
    ├── scenes[]
    │   ├── activities ★ 首选采样单元
    │   ├── props
    │   ├── narrativeProgression
    │   └── mood / videoEditingDetails
    ├── timecoded_text_to_speech ★ 音频对齐
    └── qAndA（已有 QA 对）
```

### 2.1 推荐采样层级

**推荐以 `activities` 为主采样单元**，原因：

- 有明确的 `description`（可作为 ground truth）
- 有 `timestamp.start_timestamp` / `end_timestamp`（可裁剪片段）
- 粒度适中（通常 3–30 秒），适合音频 caption

| 层级 | 优点 | 缺点 |
|------|------|------|
| `activities` | 动作描述清晰，时间边界明确 | 部分片段无语音 |
| `scenes` | 片段更长，上下文更完整 | 描述偏宏观，区分度低 |
| `props` | 物体级 caption | 很多片段视觉静态 |
| `qAndA` | 已有 question/answer | 未必带精确时间戳，需二次对齐 |

### 2.2 片段过滤条件

- 片段时长 **3–30 s**（兼顾 Qwen3-Omni audio token 与 FineVideo 标注质量）
- `description` 非空且长度 > 10 字符
- 可选：该时间段内 `timecoded_text_to_speech` 有文本（保证有语音）
- 可选：过滤 `audioVisualCorrelation` 过低的 scene

---

## 3. Qwen3-Omni 输入约束

### 3.1 支持的模态与消息格式

Qwen3-Omni 通过 `qwen_omni_utils.process_mm_info()` 解析 chat messages，**四种模态彼此独立**：

| 模态 | message 写法 | 输入形态 |
|------|-------------|---------|
| 音频 | `{"type": "audio", "audio": "<path/url>"}` | WAV/MP3 等，**内部重采样到 16 kHz** |
| 图像 | `{"type": "image", "image": "<path/url>"}` | 单张静态图 |
| 视频 | `{"type": "video", "video": "<path/url>"}` | **完整 mp4**，由工具链抽帧 |
| 文本 | `{"type": "text", "text": "..."}` | 问题/指令 |

官方明确支持 **audio + image 混合输入（不含 video）**：

```python
{"type": "image", "image": "cars.jpg"},
{"type": "audio", "audio": "asr_fr.wav"},
{"type": "text", "text": "What can you see and hear?"}
```

这与「独立音频 + 抽帧图像、不传整段视频」的业务形态 **完全吻合**。

### 3.2 关键参数：`use_audio_in_video`

| 场景 | 设置 | 说明 |
|------|------|------|
| 输入 **mp4 视频**，希望模型同时听视频内音轨 | `use_audio_in_video=True` | 从视频中自动提取音频，并与视觉 temporal ID 对齐（80 ms 粒度） |
| 输入 **独立 WAV + 多张 image**（本方案） | `use_audio_in_video=False` | 必须关闭，否则 processor 行为不符合预期 |

注意：

- 多轮对话中该参数 **必须全程一致**
- **vLLM-Omni rollout 不支持** `use_audio_in_video`；需分别传 `audio` 与 `image`（不传 video）
- 若 `use_audio_in_video=True`，**audio 与 video 数量必须相等**（本方案不走此路径）

**结论：FineVideo 预处理应走「WAV + N 张 image」路径，全程 `use_audio_in_video=False`。这与 verl-omni 选用 vLLM-Omni 作为 Qwen3-Omni rollout backend 的约束一致。**

### 3.3 每 prompt 模态数量上限（vLLM-Omni）

官方默认示例：

```python
limit_mm_per_prompt={'image': 3, 'video': 3, 'audio': 3}
```

本方案需要 **1 路 audio + 5~6 张 image**，推理时必须调整为：

```python
limit_mm_per_prompt={'image': 6, 'audio': 1, 'video': 0}
```

否则超出限制的 image 会被截断或报错。

### 3.4 分辨率 / 像素约束

Qwen3-Omni 视觉编码沿 Qwen3-VL 体系，默认：

- `min_pixels = 128 × 28 × 28 = 100,352`
- `max_pixels = 768 × 28 × 28 = 602,112`（约 768 visual tokens/帧）

FineVideo 分辨率多为 **640×360 = 230,400 px**，在默认范围内，**无需额外缩放**。

可在每张 image 上显式指定（推荐写入数据集 `extra_info` 以便复现）：

```python
{"type": "image", "image": "frame_00.jpg",
 "min_pixels": 128*28*28, "max_pixels": 768*28*28}
```

### 3.5 音频处理约束

- 内部：**16 kHz 单声道** → 128 通道 mel-spectrogram（25 ms 窗、10 ms hop）
- 每帧 audio representation ≈ **80 ms** 原始信号
- 预处理建议：`ffmpeg -vn -ac 1 -ar 16000` 导出 WAV（与模型一致，减少运行时开销）
- 需要系统安装 **ffmpeg**（`qwen_omni_utils` 依赖）
- 片段建议 **2–30 秒**；过长会占用大量 context（`max_model_len` 通常 32768）

### 3.6 视频 vs 多图：为何不把抽帧当 video 传

| 方案 | Qwen3-Omni 行为 | 是否适合本任务 |
|------|----------------|---------------|
| 传 **短 mp4** + `use_audio_in_video=True` | 动态 fps 抽帧 + 80 ms temporal ID + 音画对齐 | 功能最强，但不符合「不传视频」的业务要求 |
| 传 **6 张 image + 1 路 audio** + `use_audio_in_video=False` | 每帧独立编码，**无视频 temporal ID** | **符合业务**，适合 caption / 选择题型 |
| 把抽帧拼成 mp4 再传 video | 多此一举，且与独立 WAV 重复 | 不推荐 |

**要点**：OpenCV 抽帧后存为 **6 个独立 image 文件**，在 prompt 里逐个 `{type: image}` 引用；**不要**存 mp4，也**不要**只放一个 `images` 列表字段而不写进 message content。

---

## 4. 整体构建流程

```
1. Streaming 加载 FineVideo
       ↓
2. 解析 json，提取 activity 片段候选
       ↓
3. ffmpeg 裁剪 WAV（16 kHz mono）
   OpenCV 均匀抽 6 帧 → JPG
       ↓
4. 构造 Qwen3-Omni messages（audio + images + text）
       ↓
5. 构造正负样本
       ↓
6. 写入 parquet + 媒体 sidecar 目录
       ↓
7. verl-omni rollout（vLLM-Omni）加载 parquet，use_audio_in_video=False
```

### 4.1 采样伪代码

```python
for sample in load_dataset("HuggingFaceFV/finevideo", streaming=True):
    meta = sample["json"]
    video_bytes = sample["mp4"]
    video_id = meta["original_json_filename"].replace(".json", "")

    for scene in meta["content_metadata"]["scenes"]:
        for activity in scene["activities"]:
            start, end = parse_ts(activity["timestamp"])
            duration = end - start
            if duration < 3.0 or duration > 30.0:
                continue
            yield SegmentCandidate(
                video_id=video_id,
                description=activity["description"],
                start=start,
                end=end,
                scene_title=scene["title"],
                category=meta["content_parent_category"],
            )
```

### 4.2 媒体预处理

```bash
# 音频：16 kHz mono，与模型内部一致
ffmpeg -ss {start} -to {end} -i segment_src.mp4 \
  -vn -ac 1 -ar 16000 -y audio.wav

# 视频帧：OpenCV 在 [start, end] 内均匀取 6 帧，保存 RGB JPG
# frame_00.jpg ... frame_05.jpg（按时间顺序命名）
```

**不再产出 `segment.mp4`**（除非另有用途）。

### 4.3 目录结构

```
finevideo_processed/
├── train/
│   ├── {video_id}/
│   │   ├── act_003/
│   │   │   ├── audio.wav
│   │   │   ├── frame_00.jpg
│   │   │   ├── ...
│   │   │   ├── frame_05.jpg
│   │   │   └── meta.json
├── test/
│   └── ...
├── train.parquet
└── test.parquet
```

**不要在 parquet 里存 mp4 bytes**，只存相对路径（与 verl-omni 现有 OCR/DPO 数据一致）。

---

## 5. 问题与答案构造

任务描述是「依据问题选取 description 作为答案」，可有两种形式。

> **语言约束**：写入 parquet 的 `prompt` 中所有 **question / instruction 文本必须为英文**。FineVideo 的 `description`、`qAndA` 等标注本身为英文，可直接作为 answer / candidate；**禁止**在 parquet 问题字段中使用中文。

### 5.1 生成式（caption）

```
Question: Based on the audio and the images above (shown in chronological order), describe what is happening in this clip.
Answer:   {activity.description}  ← ground_truth（英文，来自 FineVideo）
```

### 5.2 选择题（更易构造负样本）

```
Question: Which of the following best describes the audio-visual content in this clip?
A. {correct description}
B. {hard negative: other activity from same video}
C. {easy negative: random description from another video}
D. {distractor: related to scene title but incorrect}
Answer: A
```

### 5.3 英文问题模板

预处理脚本中建议固定使用英文模板（可随机轮换以增加多样性）：

| 模板 ID | 英文问题模板 |
|---------|-------------|
| `caption_generic` | `Based on the audio and the images above (shown in chronological order), describe what is happening in this clip.` |
| `caption_scene` | `In the scene titled "{scene_title}", what is the main activity shown in the audio and images above?` |
| `mcq_generic` | `Which of the following best describes the audio-visual content in this clip?\nA. ...\nB. ...\nC. ...\nD. ...` |
| `mcq_activity` | `What is happening in this clip?\nA. ...\nB. ...\nC. ...\nD. ...` |
| `finevideo_qa` | 直接复用 FineVideo `qAndA[].question`（已为英文），answer 对齐到 activity `description` |

### 5.4 预处理脚本中的语言检查

```python
ENGLISH_QUESTION_TEMPLATES = [
    "Based on the audio and the images above (shown in chronological order), describe what is happening in this clip.",
    "Which of the following best describes the audio-visual content in this clip?",
    # ...
]

def build_question(template_id: str, **kwargs) -> str:
    """All returned questions must be English."""
    ...
```

导出 parquet 前可选校验：question 文本不含 CJK 字符（`\u4e00-\u9fff` 等）。

---

## 6. 正负样本构建

| 负样本类型 | 构造方式 | 难度 |
|------------|----------|------|
| 同视频 hard negative | 同视频其他 activity 的 description | 高（推荐） |
| 同 category negative | 同 `content_parent_category` 其他片段 | 中 |
| 随机 negative | 任意其他视频 description | 低 |
| 文本扰动 negative | 替换人物名/动作词 | 中 |

### DPO / preference 格式

```json
{
  "answer_win": "Sara stands in front of a sign and talks about the course.",
  "answer_lose": "Bill emphasizes high employment rates in an office setting.",
  "win_score": 1.0,
  "lose_score": 0.0
}
```

正负样本 **共用同一组 audio + images**。

---

## 7. 业务 Parquet Schema（Qwen3-Omni 对齐）

```json
{
  "data_source": "finevideo/audio_caption",
  "ability": "audio_caption",

  "prompt": [
    {
      "role": "user",
      "content": [
        {"type": "audio", "audio": "train/{vid}/act_003/audio.wav"},
        {"type": "image", "image": "train/{vid}/act_003/frame_00.jpg",
         "min_pixels": 100352, "max_pixels": 602112},
        {"type": "image", "image": "train/{vid}/act_003/frame_01.jpg",
         "min_pixels": 100352, "max_pixels": 602112},
        {"type": "image", "image": "train/{vid}/act_003/frame_02.jpg",
         "min_pixels": 100352, "max_pixels": 602112},
        {"type": "image", "image": "train/{vid}/act_003/frame_03.jpg",
         "min_pixels": 100352, "max_pixels": 602112},
        {"type": "image", "image": "train/{vid}/act_003/frame_04.jpg",
         "min_pixels": 100352, "max_pixels": 602112},
        {"type": "image", "image": "train/{vid}/act_003/frame_05.jpg",
         "min_pixels": 100352, "max_pixels": 602112},
        {"type": "text", "text": "The images above are shown in chronological order. Which of the following best describes the audio-visual content in this clip?\nA. ...\nB. ...\nC. ...\nD. ..."}
      ]
    }
  ],

  "images": [
    "train/{vid}/act_003/frame_00.jpg",
    "train/{vid}/act_003/frame_01.jpg",
    "train/{vid}/act_003/frame_02.jpg",
    "train/{vid}/act_003/frame_03.jpg",
    "train/{vid}/act_003/frame_04.jpg",
    "train/{vid}/act_003/frame_05.jpg"
  ],
  "audios": ["train/{vid}/act_003/audio.wav"],

  "mm_processor_kwargs": {
    "use_audio_in_video": false
  },

  "reward_model": {
    "style": "rule",
    "ground_truth": "A"
  },

  "extra_info": {
    "video_id": "d6b4OmUFt7I",
    "segment_id": "act_003",
    "segment_type": "activity",
    "start_timestamp": "00:00:00.000",
    "end_timestamp": "00:00:09.009",
    "scene_title": "Introductory Scenes",
    "description": "Sara stands in front of a sign and talks about the course.",
    "num_frames": 6,
    "frame_order": ["frame_00.jpg", "frame_01.jpg", "frame_02.jpg", "frame_03.jpg", "frame_04.jpg", "frame_05.jpg"],
    "candidates": ["...", "...", "...", "..."],
    "negative_descriptions": ["...", "..."],
    "split": "train",
    "index": 42
  }
}
```

### 与 verl-omni 配置的对应关系

| 字段 | 说明 |
|------|------|
| `prompt_key: prompt` | chat 格式消息 |
| `image_key: images` | 帧路径列表（`legacy_data.yaml` 默认 `images`） |
| `audios` | 音频路径列表；若无 `audio_key`，由 custom dataset 从 prompt 或 extra_info 读取 |
| `mm_processor_kwargs.use_audio_in_video` | 固定 `false`（vLLM-Omni rollout 分别传 audio + image） |
| `reward_model.ground_truth` | 正确答案（生成式为英文 description，选择题为选项字母或完整 description） |
| **question 语言** | **必须英文**；写入 `prompt` 的 `{type:text}` 字段 |
| **rollout backend** | verl-omni 使用 **`vllm_omni`**，见第 8 节 |

### 与初版方案的主要差异

| 初版 | 修正后（Qwen3-Omni 对齐） |
|------|--------------------------|
| 可能有 `audio_path` 单字段 | 增加 `audios` 列表 + message 内 `{type:audio}` |
| 6 帧只放 `images` 字段 | **必须在 `prompt.content` 中逐个 `{type:image}` 声明** |
| 未提 `use_audio_in_video` | 固定 `false`，写入 `mm_processor_kwargs` |
| 未提像素约束 | 每帧带 `min_pixels` / `max_pixels` |
| 可能保留 mp4 | **不产出/不传 mp4** |

---

## 8. verl-omni Rollout：vLLM-Omni Backend

Qwen3-Omni 在 verl-omni 中的 **rollout backend 选用 vLLM-Omni**（与 Qwen-Image 等扩散模型 rollout 共用 `vLLMOmniHttpServer` 基础设施，但模型类型为 Qwen3-Omni Thinker 文本生成）。

### 8.1 训练配置要点

```yaml
# actor_rollout_ref.rollout.name 设为 vllm_omni
actor_rollout_ref:
  rollout:
    name: vllm_omni
    # vLLM-Omni engine 参数（示意，以实际 trainer yaml 为准）
    limit_mm_per_prompt:
      image: 6
      audio: 1
      video: 0
    max_model_len: 32768
```

与现有 FlowGRPO 示例一致，rollout 引擎名称为 `vllm_omni`（参见 `examples/flowgrpo_trainer/run_qwen_image_ocr*.sh` 中的 `ENGINE=vllm_omni` 模式）。

### 8.2 数据集 → rollout 的数据流

```
parquet row
  prompt (English question + audio/image paths in content)
  audios / images
  mm_processor_kwargs: {use_audio_in_video: false}
       ↓
AgentLoop: process_mm_info(messages, use_audio_in_video=False)
       ↓
vLLM-Omni generate
  multi_modal_data: {image: [...], audio: [...]}
  mm_processor_kwargs: {use_audio_in_video: False}
```

**关键约束（vLLM-Omni 与 Transformers 差异）**：

| 项目 | 要求 |
|------|------|
| Rollout backend | `vllm_omni`，不用纯 Transformers 在线 rollout |
| `use_audio_in_video` | **不可用**；分别传 `audio` + `image` |
| `limit_mm_per_prompt` | 至少 `image: 6, audio: 1, video: 0` |
| 本地媒体路径 | vLLM serve 需 `--allowed-local-media-path` 覆盖数据目录 |
| 问题语言 | 英文（与 Qwen3-Omni 预训练/评测语料一致） |

### 8.3 vLLM-Omni 推理请求格式（rollout 对齐）

Dataset collate / agent loop 最终应产出与下列结构等价的请求（`qwen_omni_utils.process_mm_info` 解析 parquet 中的 messages）：

```python
from qwen_omni_utils import process_mm_info
from transformers import Qwen3OmniMoeProcessor

processor = Qwen3OmniMoeProcessor.from_pretrained("Qwen/Qwen3-Omni-30B-A3B-Thinking")

messages = row["prompt"]  # user content: audio + 6 images + English text

text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
audios, images, videos = process_mm_info(messages, use_audio_in_video=False)

inputs = {
    "prompt": text,
    "multi_modal_data": {},
    "mm_processor_kwargs": {"use_audio_in_video": False},
}
if images is not None:
    inputs["multi_modal_data"]["image"] = images
if audios is not None:
    inputs["multi_modal_data"]["audio"] = audios

outputs = llm.generate([inputs], sampling_params=sampling_params)
```

### 8.4 Transformers（仅离线调试 / smoke test）

正式 RL rollout 走 vLLM-Omni；Transformers 仅用于预处理验证或单机 debug：

```python
inputs = processor(
    text=text, audio=audios, images=images, videos=videos,
    return_tensors="pt", padding=True, use_audio_in_video=False,
)
```

**内容顺序建议**：**audio → images（时间序）→ English text**，与官方 mixed-media 示例一致。

---

## 9. 模型选型建议

| 任务 | 推荐模型 |
|------|---------|
| 音视频理解 + 选择题/推理 | `Qwen3-Omni-30B-A3B-Thinking` |
| 纯音频 caption（无画面） | `Qwen3-Omni-30B-A3B-Captioner` |
| 本任务（audio + 6 frames） | **Thinking 或 Instruct（disable talker）** |

> `Qwen3-Omni-30B-A3B-Captioner` **只接受 audio**，不适合 audio+image 任务。

---

## 10. 数据量与划分建议

| 项目 | 建议 |
|------|------|
| 试点规模 | 先处理 500–2000 个 activity 片段验证 pipeline |
| 训练集 | 按 **video_id** 划分（同视频所有 activity 进同一 split，防泄漏） |
| 测试集 | 10–15% video，或固定 500 条 |
| 每视频采样 | 每视频随机取 1–3 个 activity，避免长尾视频主导 |
| 许可 | CC-BY，需保留 provenance（`youtube_channel`, `video_id` 写入 `extra_info`） |

预估规模：43k 视频 × 平均 ~5–10 activities ≈ **20–40 万候选片段**，过滤后约 **10–20 万** 可用样本。

全量约 600GB，务必用 `streaming=True`，只下载需要的 mp4。

---

## 11. 风险与缓解

| 风险 | 缓解措施 |
|------|---------|
| 6 张 image 无 video temporal ID | 帧按时间顺序排列；英文 prompt 注明 `shown in chronological order`；片段控制在 30 s 内 |
| vLLM-Omni `limit_mm_per_prompt` 默认过小 | rollout yaml 显式设为 `image: 6, audio: 1, video: 0` |
| parquet 问题含中文 | 预处理固定英文模板；导出前 CJK 校验 |
| GPU OOM | 优先减帧（6→4）或降低 `max_pixels`，而非减 audio |
| 音频与 description 不对齐 | 用 `timecoded_text_to_speech` 检查该时间段是否有 speech |
| 抽帧与音频时间不对齐 | 帧和音频必须从同一 `[start, end]` 裁剪 |
| qAndA 复用 | 需验证 answer 是否对应某个 activity 的 description |

---

## 12. 推荐实施顺序

1. **Streaming 下载 50 条**，验证 timestamp 解析 + ffmpeg/OpenCV 提取
2. **定义最终 schema**（与 Qwen3-Omni audio/image 输入接口对齐）
3. **写 `finevideo_qwen3_omni.py` 预处理脚本**（类似 `examples/flowgrpo_trainer/data_process/qwenimage_ocr.py`）
4. **小规模人工抽检** 50 条：听音频、看帧、核对 description
5. **批量跑全量**，输出 `train.parquet` / `test.parquet`
6. **构造负样本**（同视频 hard negative 优先）
7. **vLLM-Omni smoke test**：`rollout.name=vllm_omni`，确认 `limit_mm_per_prompt` 与 `use_audio_in_video=False`
8. **英文问题抽检**：确认 parquet 中无中文 question

---

## 13. 小结

FineVideo 非常适合本任务：它的 `activities` 提供了 **带时间戳的 description**，`timecoded_text_to_speech` 支持音频对齐，`qAndA` 可辅助问题设计。

核心改造是：

> **视频级 → 片段级；mp4 → `{audio.wav, frame_*.jpg}`；嵌套 JSON → 扁平 parquet 行**

按 Qwen3-Omni + verl-omni 对齐的关键约束：

1. **不传 video**；6 帧作为 **6 个独立 `{type:image}`**，音频作为 **`{type:audio}`**
2. 全程 **`use_audio_in_video=False`**（vLLM-Omni rollout 必须分别传 audio + image）
3. Rollout 使用 **vLLM-Omni**（`rollout.name=vllm_omni`），**`limit_mm_per_prompt={'image': 6, 'audio': 1, 'video': 0}`**
4. parquet 中所有 **question / instruction 使用英文**；answer / candidate 沿用 FineVideo 英文 `description`
5. 帧分辨率落在 **128~768 × 28²** 像素范围内

---

## 参考资料

- [FineVideo Dataset](https://huggingface.co/datasets/HuggingFaceFV/finevideo)
- [Qwen3-Omni GitHub](https://github.com/QwenLM/Qwen3-Omni)
- [Qwen3-Omni HuggingFace Model Card](https://huggingface.co/Qwen/Qwen3-Omni-30B-A3B-Instruct)
- [Qwen3-Omni Technical Report](https://arxiv.org/html/2509.17765v1)
- [vLLM-Omni Qwen3-Omni 推理文档](https://docs.vllm.ai/projects/vllm-omni/en/latest/user_guide/examples/offline_inference/qwen3_omni/)

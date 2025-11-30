**LLM+VQA Checklist（看哪些文件）**

* 入口脚本/命令：**tools/test.py**, **tools/train.py**（看如何加载 config/args/launcher），**run_test.sh**(如有)。
* Config 入口：**adzoo/orion/configs/orion_stage3_infer_llm_vqa_light.py**, **adzoo/orion/configs/orion_stage3_infer.py**, **adzoo/orion/configs/orion_stage3_fp16.py**, **adzoo/orion/configs/orion_stage3_cot.py**, 极简禁LLM参考 **adzoo/orion/configs/orion_stage1_train_minimal.py**。
* 数据集 & pipeline：**mmcv/datasets/b2d_orion_dataset.py**, **mmcv/datasets/vis_utils.py**(可视化/调试), 数据加载/采样在 **adzoo/orion/configs/_base_/datasets/nus-3d.py** 及各 stage config 的 **train_pipeline**/**test_pipeline**/**inference_only_pipeline**。
* 模型总装 Orion：**adzoo/orion/models/orion.py**(主模型), 各子模块 **adzoo/orion/models/heads**、**.../detectors**、**.../transformer**，以及 **OrionHead/OrionHeadM** 定义。
* LLM 加载 & VQA 前向：查 **LoadAnnoatationCriticalVQATest**/**vlm_labels** 的数据处理（**mmcv/datasets** 下相关 transforms），LLM/Q-Former 在 **ckpts/pretrain_qformer/** 路径配置，模型里 tokenizer/lm_head 的处理。
* 推理/评测流程：**tools/test.py** 中 eval 流程，**orion_stage*_config** 的 **log_config**、**evaluation** 段落；若有 demo/agent：**adzoo/orion/configs/orion_stage3_agent.py**；可参考 **orion_stage3_train.py** 里的 **inference_only_pipeline**。

**3 天速通路线**

* 第 1 天：读 config 与入口
  * 通读 **orion_stage3_infer_llm_vqa_light.py**→**orion_stage3_infer.py** 对比差异，弄清 VQA 开关（**LoadAnnoatationCriticalVQATest**、**use_gen_token**、**max_length**、**desc_qa**）、LLM 路径、fp16/内存设置。
  * 看 **tools/test.py** 参数入口，跑一次干测（不下权重也行）确认 pipeline 走通。
* 第 2 天：梳理数据与 pipeline
  * 读 **mmcv/datasets/b2d_orion_dataset.py** 和 transforms（含 **LoadAnnoatationCriticalVQATest**）明确输入张量/label/**input_ids**/**vlm_labels** 的打包。
  * 查 **_base_/datasets/nus-3d.py** 及 stage3 config 的 pipeline，画出数据流：文件路径→图像增强→VQA token→Collect。
  * 小批量验证：用 **--show**/可视化函数（**vis_utils.py**）检查 VQA 字段是否存在。
* 第 3 天：模型与前向
  * 读 **adzoo/orion/models/orion.py**，定位 tokenizer/lm_head 挂载、**use_gen_token**、**use_col_loss**、**use_memory** 对 forward 的影响。
  * 读 **OrionHead**/**OrionHeadM** 和 transformer 解码器，确认 VQA token 如何参与注意力。
  * 结合 **orion_stage3_fp16.py**/**orion_stage3_cot.py** 看大 batch/完整配置，收束到轻量版参数，准备正式推理/评测。

**关键字索引（搜这些快速定位）**

* 在 configs：**LoadAnnoatationCriticalVQATest**, **use_gen_token**, **max_length**, **desc_qa**, **tokenizer**, **lm_head**, **fp16_infer**, **use_memory**, **memory_len**, **collect_keys**, **inference_only_pipeline**.
* 在数据集/transform：**input_ids**, **vlm_labels**, **critical_qa**, **CustomCollect3D**, **PETRFormatBundle3D**.
* 在模型：**class Orion(**, **forward_vqa**, **apply_lora**, **fp16_eval**, **use_col_loss**, **use_diff_decoder**, **memory_decoder**.
* 在脚本：**parse_args** in **tools/test.py**, **build_dataloader**, **build_dataset**, **single_gpu_test**/**multi_gpu_test**.

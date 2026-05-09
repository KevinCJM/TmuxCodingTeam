# -*- encoding: utf-8 -*-
"""
@File: Prompt_06_TaskSplit.py
@Modify Time: 2026/4/10 15:08       
@Author: Kevin-Chen
@Descriptions: 任务拆分阶段提示词
"""

from tmux_core.prompt_contracts.spec import (
    ACCESS_READ,
    ACCESS_READ_WRITE,
    ACCESS_WRITE,
    CHANGE_MUST_CHANGE,
    CHANGE_NONE,
    CLEANUP_NONE,
    CLEANUP_SYSTEM_BEFORE_STAGE_OR_RETRY,
    SPECIAL_OPEN_HITL,
    SPECIAL_REVIEW_FAIL,
    SPECIAL_REVIEW_PASS,
    SPECIAL_STAGE_ARTIFACT,
    FileSpec,
    OutcomeSpec,
    agent_prompt,
)
from T04_common_prompt import task_start_prompt, state_machine_output, main_agent_workflow_after_review


# [需求分析师] 生成任务单
@agent_prompt(
    prompt_id="a06.task_split.generate",
    stage="a06",
    role="requirements_analyst",
    intent="generate_task_split",
    mode="a06_task_split_generate",
    files={
        "task_md": FileSpec(
            path_arg="task_md",
            access=ACCESS_WRITE,
            change=CHANGE_MUST_CHANGE,
            meaning="Markdown 任务单",
            cleanup=CLEANUP_SYSTEM_BEFORE_STAGE_OR_RETRY,
            special=SPECIAL_STAGE_ARTIFACT,
        ),
        "detailed_design": FileSpec(path_arg="detail_design_md", access=ACCESS_READ, change=CHANGE_NONE),
        "hitl_record": FileSpec(path_arg="hitl_record_md", access=ACCESS_READ, change=CHANGE_NONE),
        "original_requirement": FileSpec(path_arg="original_requirement_md", access=ACCESS_READ, change=CHANGE_NONE),
        "requirements_clear": FileSpec(path_arg="requirements_clear_md", access=ACCESS_READ, change=CHANGE_NONE),
    },
    outcomes={"completed": OutcomeSpec(status="completed", requires=("task_md",))},
)
def task_split(task_md='name_任务单.md', detail_design_md='name_详细设计.md', hitl_record_md='name_人机交互澄清记录.md',
               original_requirement_md='name_原始需求.md', requirements_clear_md='name_需求澄清.md'):
    task_split_prompt = f"""# Role: 高级技术产品经理 (Technical PM) / 交付架构师

## Context
你负责将已完成的《{detail_design_md}》转化为可执行的工程交付计划。
- **输入依据**：核心逻辑必须严格对齐 《{original_requirement_md}》+《{requirements_clear_md}》+《{hitl_record_md}》。
- **拆解对象**：《{detail_design_md}》中定义的所有功能模块、技术组件及异常处理路径。
- **输出目标**：生成极具实操性的《{task_md}》，作为后续开发 Agent 的直接指令集。

## Task Structure: 里程碑 (Milestone) & 任务单 (Task)
你必须严格遵循“两级拆解协议”，确保研发进度的可视化与原子化：

### 1. 里程碑 (M) —— 阶段性交付目标
- **定义**：具有业务价值的阶段性终点（例如：核心模型就绪、API 链路跑通、UI 交互完整）。
- **要求**：里程碑必须是“可演示”或“可集成”的状态。

### 2. 任务单 (T) —— 最小可执行单元
- **原子性**：单任务开发周期建议在 2-4 小时逻辑内，严禁包含跨模块的复杂任务。
- **独立性**：任务之间应具备清晰的边界，前置依赖需在标题或目标中体现。
- **闭环性**：每个任务必须自带“验收标准”，拒绝“完成某功能”这种模糊描述。

## Task Template (Mandatory)
- 请严格按此 Markdown 格式输出：
```md
## 里程碑 M[序号]: <明确的交付成果名称>
- **状态**: 未开始
- **阶段性成果**: <描述该阶段完成后，系统具备了什么新能力>
- **完成判定**: <业务层面的硬性验收指标>
- **任务清单**:
  - [ ] M[序号]-T[序号] <任务简述>
    - **目标**: <具体的逻辑实现点，需引用 `代码标识符` 或 `接口名`>
    - **涉及**: <具体的目录、文件名或数据库表>
    - **完成标准**: <定义非黑即白的完成状态，如：返回值符合 Schema、日志覆盖异常路径>
    - **验证方式**: <具体的测试指令、SQL 查询或 API 调用示例>
```
- 里程碑编号要求格式如 `M1`, `M2`, 具体任务编号要求格式如 `M1-T1`, `M2-T1`。

## Engineering Constraints (Strong)
1. **禁止混淆**：里程碑是“状态”，任务单是“动作”。禁止出现“完成 M1”这种任务单。
2. **逻辑顺序**：任务单必须按开发拓扑排序（如：先建表 -> 后写 Model -> 再写 Service -> 最后挂载 Controller）。
3. **异常对齐**：必须包含《{detail_design_md}》中提到的异常处理、边界判定和日志埋点任务。
4. **拒绝冗余**：不服务于需求实现的“重构”或“技术债优化”严禁入单，除非详细设计中明确要求。
5. **四问校验**: 禁止为了最求代码简洁优雅而修改代码,禁止为了解决原代码技术债而修改代码,禁止主动修复自以为的BUG,在改动代码前必须回答：
    * 是否属于当前需求？如果不是，不改。
    * 不改是否阻塞？如果不会阻塞，不改。
    * 是否可以最小改造？避免全面重构，而是只解决当前障碍，只做最小化的改动。
    * 是否影响原有契约？禁止影响原有契约，包括 API、配置、日志、数据结构、下游依赖、用户工作流、兼容等等逻辑。

## Workflow
1. 扫描所有输入文档，识别核心路径与依赖。
2. 划分 3-10 个逻辑合理的里程碑。
3. 针对每个里程碑，进行原子化任务拆解。
4. 进行自检：是否每个任务单拿出来，一个初级开发都能在不看原需求的情况下直接开工？

## 约束
- 禁止修改除了《{task_md}》以外的任何文档与源代码;
- 直接返回字符串 `完成`，禁止返回任何其他文字。"""
    return task_split_prompt


# 初始化智能体
@agent_prompt(
    prompt_id="a06.task_split.ba_init",
    stage="a06",
    role="requirements_analyst",
    intent="ready",
    mode="a06_ba_init",
    files={
        "detailed_design": FileSpec(path_arg="detail_design_md", access=ACCESS_READ, change=CHANGE_NONE),
        "hitl_record": FileSpec(path_arg="hitl_record_md", access=ACCESS_READ, change=CHANGE_NONE),
        "original_requirement": FileSpec(path_arg="original_requirement_md", access=ACCESS_READ, change=CHANGE_NONE),
        "requirements_clear": FileSpec(path_arg="requirements_clear_md", access=ACCESS_READ, change=CHANGE_NONE),
    },
    outcomes={"ready": OutcomeSpec(status="ready")},
)
def create_task_split_ba(ba_desc, init_prompt=task_start_prompt,
                         detail_design_md='name_详细设计.md', hitl_record_md='name_人机交互澄清记录.md',
                         original_requirement_md='name_原始需求.md', requirements_clear_md='name_需求澄清.md'):
    detailed_design_prompt = f"""## 角色定位
{ba_desc}

## 任务
* 基于《{requirements_clear_md}》+《{original_requirement_md}》+《{hitl_record_md}》+《{detail_design_md}》理解当前需求和详细设计。
* 基于代码现状, 理解需求与代码直接的对应关系。
* 梳理要完成该需求所需的代码改造点, 以及各个改造点之间的前后关系。
* 本次任务仅仅做理解, 不要修改任何源代码或文档。

## 约束
- 不要输出理解结论或其他额外说明。
- 仅输出 `完成`, 不要输出其他文本。
- 禁止修改任何源代码或文档。

---

{init_prompt}"""
    return detailed_design_prompt


# [审核员] 审核任务单安排
@agent_prompt(
    prompt_id="a06.task_split.reviewer_round",
    stage="a06",
    role="reviewer",
    intent="review",
    mode="a06_reviewer_round",
    files={
        "original_requirement": FileSpec(path_arg="original_requirement_md", access=ACCESS_READ, change=CHANGE_NONE),
        "requirements_clear": FileSpec(path_arg="requirements_clear_md", access=ACCESS_READ, change=CHANGE_NONE),
        "hitl_record": FileSpec(path_arg="hitl_record_md", access=ACCESS_READ, change=CHANGE_NONE),
        "task_md": FileSpec(path_arg="task_md", access=ACCESS_READ, change=CHANGE_NONE),
        "detailed_design": FileSpec(path_arg="detail_design_md", access=ACCESS_READ, change=CHANGE_NONE),
        "review_md": FileSpec(
            path_arg="task_review_md",
            access=ACCESS_WRITE,
            change=CHANGE_MUST_CHANGE,
            meaning="任务拆分未通过时的评审问题",
            cleanup=CLEANUP_SYSTEM_BEFORE_STAGE_OR_RETRY,
        ),
        "review_json": FileSpec(
            path_arg="task_review_json",
            access=ACCESS_READ_WRITE,
            change=CHANGE_MUST_CHANGE,
            meaning="任务拆分评审 pass/fail 结构化事实源",
            cleanup=CLEANUP_SYSTEM_BEFORE_STAGE_OR_RETRY,
        ),
    },
    outcomes={
        "review_pass": OutcomeSpec(status="review_pass", requires=("review_json",), forbids=("review_md",), special=SPECIAL_REVIEW_PASS),
        "review_fail": OutcomeSpec(status="review_fail", requires=("review_json", "review_md"), special=SPECIAL_REVIEW_FAIL),
    },
)
def review_task(agent_desc, task_name='任务拆分', *, original_requirement_md='name_原始需求.md',
                requirements_clear_md='name_需求澄清.md', hitl_record_md='name_人机交互澄清记录.md',
                task_md='name_任务单.md', detail_design_md='name_详细设计.md',
                task_review_md='name_任务单评审记录_agent.md', task_review_json='name_评审记录_agent.json'):
    state_machine_output_prompt = state_machine_output(
        task_name, task_review_md, task_review_json,
        pass_condition="任务与里程碑设计完全符合需求与详细设计, 符合研发逻辑与拓扑顺序。",
        blocked_condition="发现“逻辑不符”、“拓扑错误”、“描述歧义”、“违反最小改动”、“功能遗漏”、或发现其他问题、或存在其他疑问。")
    review_task_prompt = f"""## 角色定位
{agent_desc}

## Context
你负责对《{task_md}》进行工程化可行性审计。
- **审计基准**: 《{original_requirement_md}》/《{requirements_clear_md}》/《{hitl_record_md}》/《{detail_design_md}》
- **目标文件**: 《{task_md}》

## Audit Criteria (Five-Pillars)
1. **结构对齐**: 是否严格遵循“里程碑 (Milestone) - 任务单 (Task)”两级递归结构？是否存在同级混淆？里程碑编号要求格式如 `M1`, `M2`, 具体任务编号要求格式如 `M1-T1`, `M2-T1`。
    * 要求《{task_md}》的 Markdown 格式如下：
```md
## 里程碑 M[序号]: <明确的交付成果名称>
- **状态**: 未开始
- **阶段性成果**: <描述该阶段完成后，系统具备了什么新能力>
- **完成判定**: <业务层面的硬性验收指标>
- **任务清单**:
  - [ ] M[序号]-T[序号] <任务简述>
    - **目标**: <具体的逻辑实现点，需引用 `代码标识符` 或 `接口名`>
    - **涉及**: <具体的目录、文件名或数据库表>
    - **完成标准**: <定义非黑即白的完成状态，如：返回值符合 Schema、日志覆盖异常路径>
    - **验证方式**: <具体的测试指令、SQL 查询或 API 调用示例>
```
2. **拓扑序校验**: 任务编排是否符合研发逻辑（如：[Schema -> Model -> Service -> API/UI]）？是否存在循环依赖/前置缺失/依赖前置？
3. **需求覆盖**: 是否 100% 覆盖了详细设计中的核心逻辑、异常分支及参数校验？是否有遗漏点？
4. **原子化度量**: 任务描述是否足够具体？是否包含明确的 `代码标识符`、完成标准及验证方式？是否粒度过粗无法落地？
5. **四问过滤 (Minimalism)**: 是否包含与本次需求无关的重构、扩展或非阻断性技术债修复？
    * 是否属于当前需求？如果不是，不改。
    * 不改是否阻塞？如果不会阻塞，不改。
    * 是否可以最小改造？避免全面重构，而是只解决当前障碍，只做最小化的改动。
    * 是否影响原有契约？禁止影响原有契约，包括 API、配置、日志、数据结构、下游依赖、用户工作流、兼容等等逻辑。

{state_machine_output_prompt}

## Formatting Guidelines (Internal AI Logic)
- **输出静默**: 禁止输出任何分析过程、解释说明。只能输出指定字符串。
- **禁用修改**: 严禁修改除了《{task_review_md}》/《{task_review_json}》以外的任务文档、源代码、或配置。
- **高能引用**: 必须直接引用 `代码标识符`、`方法名` 或 `里程碑 ID`。
- **去情绪化**: 你的输出将作为下一个节点修复任务单的唯一逻辑驱动，请确保信息密度最大化。"""
    return review_task_prompt


# [需求分析师] 根据评审文档优化详细设计
@agent_prompt(
    prompt_id="a06.task_split.modify",
    stage="a06",
    role="requirements_analyst",
    intent="review_feedback",
    mode="a06_task_split_feedback",
    files={
        "task_md": FileSpec(
            path_arg="task_md",
            access=ACCESS_READ_WRITE,
            change=CHANGE_MUST_CHANGE,
            meaning="评审修订后的任务单",
            cleanup=CLEANUP_NONE,
        ),
        "ba_feedback": FileSpec(
            path_arg="what_just_change",
            access=ACCESS_WRITE,
            change=CHANGE_MUST_CHANGE,
            meaning="需求分析师对任务拆分评审的修复说明",
            cleanup=CLEANUP_SYSTEM_BEFORE_STAGE_OR_RETRY,
        ),
        "ask_human": FileSpec(
            path_arg="ask_human_md",
            access=ACCESS_WRITE,
            change=CHANGE_MUST_CHANGE,
            meaning="任务拆分评审信息不足时的 HITL 问题",
            cleanup=CLEANUP_SYSTEM_BEFORE_STAGE_OR_RETRY,
            special=SPECIAL_OPEN_HITL,
        ),
    },
    outcomes={
        "completed": OutcomeSpec(status="completed", requires=("task_md", "ba_feedback"), forbids=("ask_human",)),
        "hitl": OutcomeSpec(status="hitl", requires=("ask_human",), forbids=("ba_feedback",), special=SPECIAL_OPEN_HITL),
    },
)
def modify_task(review_msg, *,
                task_md='name_任务单.md', what_just_change='name_需求分析师反馈.md',
                ask_human_md='name_与人类交流.md'):
    main_agent_workflow_after_review_prompt = main_agent_workflow_after_review(what_just_change=what_just_change)
    modify_task_prompt = f"""## 任务背景
审计员审核了任务单《{task_md}》。你需要对这些审计员提出的评审意见进行鉴定、修复，并在信息不足时向人类发起求助。

## 输入上下文
### 审计反馈原始记录 (Raw Feedback)
[REVIEW MSG START]
{review_msg}
[REVIEW MSG END]

{main_agent_workflow_after_review_prompt}

## 约束
- 禁止修改源代码, 禁止修改除了《{what_just_change}》/《{task_md}》/《{ask_human_md}》之外的文档;
- 如果信息不足，覆盖写入《{ask_human_md}》并输出 `HITL`。
- 如果可以修复，清空《{ask_human_md}》，修改《{task_md}》，写入《{what_just_change}》，输出 `修改完成`。
- 只能输出 `HITL` 或 `修改完成`。"""
    return modify_task_prompt


# [审核员] 根据优化后的详细设计再次评审
@agent_prompt(
    prompt_id="a06.task_split.re_review",
    stage="a06",
    role="reviewer",
    intent="review_reply",
    mode="a06_reviewer_round",
    files={
        "task_md": FileSpec(path_arg="task_md", access=ACCESS_READ, change=CHANGE_NONE),
        "review_md": FileSpec(
            path_arg="task_review_md",
            access=ACCESS_READ_WRITE,
            change=CHANGE_MUST_CHANGE,
            meaning="任务拆分复评未通过时的剩余问题",
            cleanup=CLEANUP_NONE,
        ),
        "review_json": FileSpec(
            path_arg="task_review_json",
            access=ACCESS_READ_WRITE,
            change=CHANGE_MUST_CHANGE,
            meaning="任务拆分复评 pass/fail 结构化事实源",
            cleanup=CLEANUP_NONE,
        ),
    },
    outcomes={
        "review_pass": OutcomeSpec(status="review_pass", requires=("review_json",), forbids=("review_md",), special=SPECIAL_REVIEW_PASS),
        "review_fail": OutcomeSpec(status="review_fail", requires=("review_json", "review_md"), special=SPECIAL_REVIEW_FAIL),
    },
)
def again_review_task(modify_summary, task_name='任务拆分', *, task_md='name_任务单.md',
                      task_review_md='name_任务单评审记录_agent.md', task_review_json='name_评审记录_agent.json'):
    state_machine_output_prompt = state_machine_output(
        task_name, task_review_md, task_review_json,
        pass_condition="任务与里程碑设计完全符合需求与详细设计, 符合研发逻辑与拓扑顺序。",
        blocked_condition="发现“逻辑不符”、“拓扑错误”、“描述歧义”、“违反最小改动”、“功能遗漏”、或发现其他问题、或存在其他疑问。")
    again_review_task_prompt = f"""## Input Context Isolation
以下为 [需求分析师] 针对上一轮评审记录的修复说明，作为本次审计的判定基准之一：

[ANALYST_FEEDBACK_START]
{modify_summary}
[ANALYST_FEEDBACK_END]

---

## Task Objective
执行“二审核销”任务。基于上述【需求分析师反馈】，对《{task_md}》进行增量审计与状态同步。

## Audit Criteria (Five-Pillars)
1. **结构对齐**: 是否严格遵循“里程碑 (Milestone) - 任务单 (Task)”两级递归结构？是否存在同级混淆？里程碑编号要求格式如 `M1`, `M2`, 具体任务编号要求格式如 `M1-T1`, `M2-T1`。
    * 要求《{task_md}》的 Markdown 格式如下：
```md
## 里程碑 M[序号]: <明确的交付成果名称>
- **状态**: 未开始
- **阶段性成果**: <描述该阶段完成后，系统具备了什么新能力>
- **完成判定**: <业务层面的硬性验收指标>
- **任务清单**:
  - [ ] M[序号]-T[序号] <任务简述>
    - **目标**: <具体的逻辑实现点，需引用 `代码标识符` 或 `接口名`>
    - **涉及**: <具体的目录、文件名或数据库表>
    - **完成标准**: <定义非黑即白的完成状态，如：返回值符合 Schema、日志覆盖异常路径>
    - **验证方式**: <具体的测试指令、SQL 查询或 API 调用示例>
```
2. **拓扑序校验**: 任务编排是否符合研发逻辑（如：[Schema -> Model -> Service -> API/UI]）？是否存在循环依赖/前置缺失/依赖前置？
3. **需求覆盖**: 是否 100% 覆盖了详细设计中的核心逻辑、异常分支及参数校验？是否有遗漏点？
4. **原子化度量**: 任务描述是否足够具体？是否包含明确的 `代码标识符`、完成标准及验证方式？是否粒度过粗无法落地？
5. **四问过滤 (Minimalism)**: 是否包含与本次需求无关的重构、扩展或非阻断性技术债修复？

## Audit & Sync Workflow
1. **核销（Remove）**：对比反馈与任务单，若原记录项已解决，从《{task_review_md}》中物理删除。
2. **追问（Update）**：若反馈声称已修复但设计文档仍有偏差，或解答引入新歧义，保留该项并追加标注 `[未结]: {{简述原因}}`。
3. **补录（Add）**：扫描设计文档全篇，若发现首轮遗漏的 [错误/遗漏/歧义/越界] 点，新增至《{task_review_md}》记录。

{state_machine_output_prompt}

## 约束
* 禁止: 禁止修改除了《{task_review_md}》/《{task_review_json}》以外的文档或代码。
* 格式: 采用极简 Bullet 形式，单行高信息密度。
* AI-to-AI 优化：你的输出不服务于人类的情绪，仅服务于后续 AI 节点的计算精度。你通过紧凑的结构提供最纯粹的逻辑差异报告，拒绝任何低信息密度的自然语言填充。
* 只能输出 `审核通过` 或 `未通过`。"""
    return again_review_task_prompt


# [需求分析师] 将任务单转为 json 格式
@agent_prompt(
    prompt_id="a06.task_split.md_to_json",
    stage="a06",
    role="requirements_analyst",
    intent="task_json",
    mode="a06_task_split_json_generate",
    files={
        "task_md": FileSpec(path_arg="task_md", access=ACCESS_READ, change=CHANGE_NONE),
        "task_json": FileSpec(
            path_arg="task_json",
            access=ACCESS_WRITE,
            change=CHANGE_MUST_CHANGE,
            meaning="任务单 JSON 状态索引",
            cleanup=CLEANUP_SYSTEM_BEFORE_STAGE_OR_RETRY,
            special=SPECIAL_STAGE_ARTIFACT,
        ),
    },
    outcomes={"completed": OutcomeSpec(status="completed", requires=("task_json",))},
)
def task_md_to_json(task_md='name_任务单.md', task_json='name_任务单.json'):
    task_md_to_json_prompt = f"""## 角色定位
你是一个【结构化数据转换专家】，负责将 Markdown 任务单转换为机器友好的状态索引JSON文件。

## 任务目标
解析《{task_md}》中的里程碑（Milestones）与任务（Tasks），提取其完成状态并写入《{task_json}》。

## 转换逻辑 (Transformation Logic)
1. **结构识别**：
   - **外层 (Key)**：提取里程碑编号（格式如 `M1`, `M2`）。
   - **内层 (Object)**：提取具体任务编号（格式如 `M1-T1`, `M2-T1`）。
2. **数据精简**：**禁止**写入任何任务描述、标题或注释。该 JSON 仅作为纯粹的状态索引。

## 输出契约 (Output Contract)
- **严格 JSON 格式**：确保《{task_json}》的内容符合严格的JSON规范。
- **禁止废话**：不提供任何分析过程或格式说明。
- **目标 Schema**：
```json
{{
  "M1": {{
    "M1-T1": false,
    "M1-T2": false
  }},
  "M2": {{
    "M2-T1": false,
    "M2-T2": false
  }}
}}
```"""
    return task_md_to_json_prompt


@agent_prompt(
    prompt_id="a06.task_split.re_md_to_json",
    stage="a06",
    role="requirements_analyst",
    intent="task_json_repair",
    mode="a06_task_split_json_generate",
    files={
        "task_md": FileSpec(path_arg="task_md", access=ACCESS_READ, change=CHANGE_NONE),
        "task_json": FileSpec(
            path_arg="task_json",
            access=ACCESS_WRITE,
            change=CHANGE_MUST_CHANGE,
            meaning="修复后的任务单 JSON 状态索引",
            cleanup=CLEANUP_SYSTEM_BEFORE_STAGE_OR_RETRY,
            special=SPECIAL_STAGE_ARTIFACT,
        ),
    },
    outcomes={"completed": OutcomeSpec(status="completed", requires=("task_json",))},
)
def re_task_md_to_json(task_md='name_任务单.md', task_json='name_任务单.json'):
    re_task_md_to_json_prompt = f"""## 问题描述
你上一轮在《{task_json}》的内容不符合要求的JSON.

## 输出契约 (Output Contract)
- **严格 JSON 格式**：确保《{task_json}》的内容符合严格的JSON规范。
- **禁止废话**：不提供任何分析过程或格式说明。
- **目标 Schema**：
```json
{{
  "M1": {{
    "M1-T1": false,
    "M1-T2": false
  }},
  "M2": {{
    "M2-T1": false,
    "M2-T2": false
  }}
}}
```

重新解析《{task_md}》覆盖输出到《{task_json}》"""
    return re_task_md_to_json_prompt


if __name__ == '__main__':
    from T04_common_prompt import check_reviewer_job
    from T01_tools import create_empty_json_files, is_standard_task_initial_json, task_done, get_markdown_content

    requirement_name = 'TimeFrequencyExtension'
    the_dir = '/Users/chenjunming/Desktop/v3_dev/tmux-api-v3'
    t_name = "任务拆分"
    agent_n_list = ['C1', 'C2']

    # 1) 生成任务单
    # print(task_split(task_md=f'{requirement_name}_任务单.md', detail_design_md=f'{requirement_name}_详细设计.md',
    #                  hitl_record_md=f'{requirement_name}_人机交互澄清记录.md',
    #                  original_requirement_md=f'{requirement_name}_原始需求.md',
    #                  requirements_clear_md=f'{requirement_name}_需求澄清.md'))

    # # 2) 审核任务单
    # a_desc = "你是一个专业的架构师, 擅长分析需求与代码直接的映射关系. 能够将复杂的代码理解为需求逻辑, 能够将复杂的需求逻辑理解为直观的代码."
    # print(review_task(a_desc, t_name,
    #                   original_requirement_md=f'{requirement_name}_原始需求.md',
    #                   requirements_clear_md=f'{requirement_name}_需求澄清.md',
    #                   hitl_record_md=f'{requirement_name}_人机交互澄清记录.md',
    #                   task_md=f'{requirement_name}_任务单.md',
    #                   detail_design_md=f'{requirement_name}_详细设计.md',
    #                   task_review_md=f'{requirement_name}_任务单评审记录_C2.md',
    #                   task_review_json=f'{requirement_name}_评审记录_C2.json'))

    # 3) 检查审核员有没有按提示词要求更新
    check_res = check_reviewer_job(agent_n_list,
                                   directory=the_dir,
                                   task_name=t_name,
                                   json_pattern=f"{requirement_name}_评审记录_*.json",
                                   md_pattern=f"{requirement_name}_任务单评审记录_*.md")

    if check_res:
        for i, m in check_res.items():
            print(i)
            print(m)
            print('-' * 100)

    # 判断是否所有评审都通过: 1)合并所有md, 2)判断总md是否为空, 3)判断所有json是否true
    pass_bool = task_done(directory=the_dir,
                          file_path=f'{the_dir}/{requirement_name}_开发前期.json',
                          task_name=t_name,
                          json_pattern=f"{requirement_name}_评审记录_*.json",
                          md_pattern=f"{requirement_name}_任务单评审记录_*.md",
                          md_output_name=f"{requirement_name}_任务单评审记录.md")

    if pass_bool:
        print(f"{t_name}阶段, 全部评审通过", '\n', '-' * 100, '\n')
        print(task_md_to_json(task_md=f'{requirement_name}_任务单.md', task_json=f'{requirement_name}_任务单.json'))

        '''判断 xx_任务单.json 是否符合要求 '''
        # if not is_standard_task_initial_json(f'{the_dir}/{requirement_name}_任务单.json'):
        #     print(re_task_md_to_json(f'{requirement_name}_任务单.md', f'{requirement_name}_任务单.json'))
    else:
        print(f"{t_name}阶段, 评审未通过", '\n', '-' * 100, '\n')

        # # 读取评审建议, 让需求分析师改
        # ba_msg = get_markdown_content(f'{the_dir}/{requirement_name}_任务单评审记录.md')
        # print(modify_task(ba_msg,
        #                   task_md=f'{requirement_name}_任务单.md', ask_human_md=f'{requirement_name}_与人类交流.md',
        #                   what_just_change=f'{requirement_name}_需求分析师反馈.md'))

        # 读取改造总结, 让审核员再次审核
        # m_summary = get_markdown_content(f'{the_dir}/{requirement_name}_需求分析师反馈.md')
        # print(again_review_task(m_summary, task_md=f'{requirement_name}_任务单.md',
        #                         task_review_md=f'{requirement_name}_任务单评审记录_C2.md',
        #                         task_review_json=f'{requirement_name}_评审记录_C2.json'))

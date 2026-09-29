"""
一个【自定义遗忘算法】的示例: BoundedGradDiff (有界梯度差分)。

动机: 原版 GradDiff 的 forget 项是 `-forget_loss` (梯度上升)，会无上限地把 forget
样本的 loss 推高，常导致模型整体崩坏。这里给 forget 的 NLL 加一个上界 tau:
一旦模型对某个 forget 样本已经"足够不确定"(NLL >= tau)，该项梯度归零，从而在遗忘
和保持模型可用性之间取得更稳的平衡。

这个例子展示了把自有算法接入本项目的最小写法:
  1. 继承一个已有的遗忘基类 (这里继承 GradDiff，白拿 retain 损失 / gamma / alpha /
     ref_model 等);
  2. 只需重写 `compute_loss` 实现你自己的 loss;
  3. 在 src/trainer/__init__.py 里 `_register_trainer(BoundedGradDiff)` 注册;
  4. 加一个 configs/trainer/BoundedGradDiff.yaml 配置 (handler + method_args)。

也可以直接继承更底层的 `UnlearnTrainer` (见 base.py)，那样只需实现 `compute_loss`，
但 retain 损失等逻辑得自己写。
"""

import torch
from trainer.unlearn.grad_diff import GradDiff


class BoundedGradDiff(GradDiff):
    def __init__(self, forget_loss_bound=4.0, *args, **kwargs):
        # method_args 里的参数会作为关键字参数传进来 (见 load_trainer)
        super().__init__(*args, **kwargs)
        self.forget_loss_bound = forget_loss_bound  # tau: forget NLL 的上界

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        # 1) forget 分支: collator 会把每个 batch 组织成 {"forget": {...}, "retain": {...}}
        forget_inputs = inputs["forget"]
        forget_inputs = {
            "input_ids": forget_inputs["input_ids"],
            "attention_mask": forget_inputs["attention_mask"],
            "labels": forget_inputs["labels"],
        }
        forget_outputs = model(**forget_inputs)
        # 有界梯度上升: NLL 一旦超过 tau，clamp 后梯度为 0，避免无限制崩坏
        bounded_nll = torch.clamp(forget_outputs.loss, max=self.forget_loss_bound)
        forget_loss = -bounded_nll

        # 2) retain 分支: 直接复用父类 GradDiff 的实现 (支持 NLL 或 KL)
        retain_inputs = inputs["retain"]
        retain_inputs = {
            "input_ids": retain_inputs["input_ids"],
            "attention_mask": retain_inputs["attention_mask"],
            "labels": retain_inputs["labels"],
        }
        retain_loss = self.compute_retain_loss(model=model, retain_inputs=retain_inputs)

        # 3) 合并: gamma / alpha 同样从父类继承
        loss = self.gamma * forget_loss + self.alpha * retain_loss
        return (loss, forget_outputs) if return_outputs else loss

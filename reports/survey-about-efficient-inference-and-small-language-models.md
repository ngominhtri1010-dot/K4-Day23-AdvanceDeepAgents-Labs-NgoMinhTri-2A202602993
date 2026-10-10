# Research Survey on Efficient Inference and Small Language Models

## TL;DR

- DUET, a collaborative inference framework, allows reasoning-intensive computation to be handled by a capable model while non–reasoning-intensive components are delegated to a lightweight model [1].
- A 4.5B parameter small language model handles multiple modalities and achieves state-of-the-art performance in multi-modal tasks [2].
- Layer-skipping methods for efficient LLM inference decide, at some granularity, which transformer layers to execute for a given input [3].

## Background

Efficient inference and small language models have gained significant attention due to the challenges posed by large-scale pre-trained models. These models are often too large for on-device computations and require substantial computational resources [4]. DistilBERT, a distilled version of BERT, demonstrates that smaller models can retain most of the language understanding capabilities of their larger counterparts, achieving a 40% reduction in size while maintaining 97% of the original performance [5]. Additionally, techniques like hardware-aware pruning, knowledge distillation, and quantization have been employed to improve the efficiency of Transformer models on CPUs [6]. The focus on efficient inference techniques aims to address issues related to model size, attention complexity, and decoding methods for large language models [7].

## Foundational Work

Knowledge distillation has been pivotal in creating smaller, faster, and more efficient language models. DistilBERT, a distilled version of BERT, retains 97% of its language understanding capabilities while reducing the model size by 40% and improving speed by 60% [5]. This approach leverages knowledge distillation during pre-training to compress the model, resulting in a smaller, faster, and lighter Transformer model [4]. Further optimizations, such as hardware-aware pruning and quantization, have enhanced the efficiency of DistilBERT on CPUs, achieving higher throughput and speedups [6]. Additionally, DistilGPT2, derived using the same distillation method, is two times faster and 33% smaller than GPT2 [8]. These methods demonstrate the effectiveness of distillation in achieving compact models without significant loss in performance.

## Recent Developments

Recent developments in efficient inference and small language models have focused on optimizing computational resources while maintaining performance. Layer-skipping methods, such as ConfLayers and SWIFT, have been rigorously audited for their efficiency in LLM inference [3]. A survey highlights the categorization of techniques to address challenges like model size and decoding methods, alongside future directions [7]. The exploration of training strategies, such as knowledge distillation and reinforcement learning, has enhanced the performance of small reasoning models [9]. ProbeScale proposes a framework combining scaling laws and probing to identify parameter-efficient subnetworks, achieving significant parameter reduction while maintaining high performance [10]. Additionally, DUET introduces a collaborative inference framework where a capable model and a lightweight model work together, reducing inference costs by up to 60% [1].

## Evaluation and Limitations

Small language models can evaluate outputs through internal representations, offering an efficient and interpretable approach [11]. A 1B-parameter model generates executable behavior trees from natural language, enabling efficient robot task planning with zero-shot capabilities [12]. Scaling laws highlight that model architecture significantly impacts inference latency, with models of the same size showing up to 3.5× latency differences [13]. The Performance-Efficiency Ratio (PER) demonstrates that small models (0.5–3B parameters) achieve superior efficiency across various tasks, outperforming larger models [14]. While wider and shallower models can improve efficiency without sacrificing accuracy, limitations persist in capturing all aspects of model performance through abstract-only evidence [13].

## Trends and open problems

Efficient inference techniques for small language models (SLMs) are actively being refined. Layer-skipping methods like ConfLayers and SWIFT show promise in balancing performance and computational efficiency [3]. Meanwhile, ProbeScale leverages scaling laws and probing analysis to identify parameter-efficient subnetworks, achieving up to 10x parameter reduction while maintaining high performance on targeted tasks [10]. The design of model architectures also plays a critical role in inference latency, with wider and shallower models offering efficiency gains without sacrificing accuracy [13]. Research continues to explore training strategies such as distillation and reinforcement learning to enhance the capabilities of SLMs [9]. Open challenges include optimizing subnetwork selection and generalizing these methods across diverse tasks and models.

## References
[1] When Less is Enough: Efficient Inference via Collaborative Reasoning. web. https://arxiv.org/html/2605.01111 (n.d.)
[2] Towards Multi-Modal Mastery: A 4.5B Parameter Truly Multi-Modal Small Language Model. hf-search. https://huggingface.co/papers/2411.05903 (2024-11-08)
[3] A rigor-matched audit of periodic-step layer skipping for efficient llm inference: conflayers versus swift, with a supplemental analysis of trained routing alternatives. arxiv. https://arxiv.org/abs/2608.28846 (2026-08-28)
[4] DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter. web. http://arxiv.org/abs/1910.01108 (2020-03-01)
[5] DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter. hf-search. https://huggingface.co/papers/1910.01108 (2019-10-02)
[6] Fast DistilBERT on CPUs. hf-search. https://huggingface.co/papers/2211.07715 (2022-10-27)
[7] A Survey on Efficient Inference for Large Language Models. hf-search. https://huggingface.co/papers/2404.14294 (2024-04-22)
[8] distillation/README.md. web. https://github.com/huggingface/transformers-research-projects/blob/main/distillation/README.md (n.d.)
[9] A Technical Study into Small Reasoning Language Models. hf-search. https://huggingface.co/papers/2506.13404 (2025-06-16)
[10] [2606.01806] ProbeScale: Probing Analysis to Optimize Neural Scaling Laws for Efficient Small Language Model Inference. web. https://arxiv.org/abs/2606.01806 (2026-06-01)
[11] Rethinking LLM-as-a-Judge: Representation-as-a-Judge with Small Language Models via Semantic Capacity Asymmetry. hf-search. https://huggingface.co/papers/2601.22588 (2026-01-30)
[12] BTGenBot-2: Efficient Behavior Tree Generation with Small Language Models. hf-search. https://huggingface.co/papers/2602.01870 (2026-02-02)
[13] Scaling Inference-Efficient Language Models. web. https://proceedings.mlr.press/v267/bian25b.html (2025-10-06)
[14] Task-Specific Efficiency Analysis: When Small Language Models Outperform Large Language. web. https://www.esann.org/sites/default/files/proceedings/2026/ES2026-274.pdf (n.d.)

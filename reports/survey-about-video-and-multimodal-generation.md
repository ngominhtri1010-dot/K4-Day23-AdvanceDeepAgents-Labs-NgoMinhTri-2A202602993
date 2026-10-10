# Survey on Video and Multimodal Generation

## TL;DR

- Video generators can realize increasingly complex visual narratives, with prompt enhancers acting as a critical bridge between user instructions and cinematic plans [1].
- VideoGPT uses VQ-VAE with 3D convolutions and axial self-attention to generate high-fidelity video samples using a GPT-like autoregressive model [2].
- VideoGPT is an architecture for scaling likelihood-based generative modeling to natural videos, using discrete latent representations and spatio-temporal position encodings [3].

## Background

Video generation has evolved significantly, with early approaches like VideoGPT [2] utilizing VQ-VAE and axial self-attention to produce high-fidelity videos. More recent methods, such as MTVG [4], incorporate diffusion-based models with advanced techniques like Dynamic Noise and Structure Guiding Sampling to ensure temporal coherence. The development of evaluation metrics has also advanced, with T2VScore [5] focusing on text-video alignment and video quality, while STREAM [6] independently assesses spatial and temporal aspects. Additionally, frameworks like WorldJen [7] introduce novel evaluation methods using Likert-scale questionnaires to provide more nuanced feedback. These developments reflect a growing emphasis on both quality and evaluation in video generation research.

## Foundational Methods

Video generation methods often rely on architectures that blend discrete latent representations with autoregressive modeling. VideoGPT uses VQ-VAE with 3D convolutions and axial self-attention to generate high-fidelity video samples using a GPT-like autoregressive model [2]. The VideoPoet framework leverages a pre-trained MAGVIT V2 video tokenizer and SoundStream audio tokenizer to transform video and audio into discrete codes compatible with language models [8]. PEBench addresses gaps in prompt enhancer evaluation by introducing a unified benchmark for direct PE assessment across text-to-video, image-to-video, and reference-to-video generation [1].

## Recent Developments

Recent developments in video and multimodal generation highlight advancements in unified frameworks and efficient evaluation methods. A unified benchmark for prompt enhancers, PEBench, enables direct evaluation across text-to-video, image-to-video, and reference-to-video tasks, addressing computational costs [1]. The MTVG method leverages diffusion-based models with dynamic noise and structure guiding for coherent video generation [4]. UniVG integrates multi-condition cross attention and biased noise for handling diverse modalities, achieving better performance on benchmarks [9]. LynnReal-Omni combines agentic visual workflows with multimodal diffusion transformers for stable, high-quality generation [10]. TV2TV introduces an interleaved text-video generation framework, improving control and quality through joint learning [11]. UnityVideo supports multi-modal, multi-task learning for world-aware video generation, enhancing zero-shot generalization [12].

## Evaluation and Limitations

Current evaluation methods for video and multimodal generation models face significant limitations. For instance, prompt enhancers are often evaluated through rendered videos, which is computationally expensive and conflates PE quality with generator behavior [1]. Additionally, existing metrics like FVD are limited to 16 frames and focus more on spatial aspects than temporal naturalness [13]. To address these issues, new metrics like T2VScore combine text-video alignment and video quality [5], while STREAM independently assesses spatial and temporal aspects of video generation [6]. Furthermore, VBench++ provides a comprehensive framework that evaluates video generation across 16 dimensions, aligning with human perception and offering insights into model strengths and weaknesses [14]. These developments highlight the need for more nuanced and human-aligned evaluation approaches in video and multimodal generation.

## Trends and open problems

Video generation research is increasingly focusing on unified frameworks that handle multiple modalities and complex visual narratives [1]. However, evaluating prompt enhancers remains challenging due to the high computational and human costs associated with rendered videos, which conflates PE quality with generator behavior [1]. Recent efforts, such as LynnReal-Omni, aim to integrate multimodal inputs and provide stable control for video generation through advanced models like 32B shared multimodal diffusion transformers [10]. Meanwhile, new evaluation metrics like STREAM and T2VScore are being developed to better assess video quality and alignment with textual descriptions [5], [6]. The need for efficient and accurate evaluation methods remains a key open problem in the field [1].

## References
[1] Towards Unified Evaluation of Prompt Enhancers for Video Generation. arxiv. https://arxiv.org/abs/2610.11736 (2026-10-08)
[2] VideoGPT: Video Generation using VQ-VAE and Transformers. hf-search. https://huggingface.co/papers/2104.10157 (2021-04-20)
[3] wilson1yan/VideoGPT. web. https://github.com/wilson1yan/videogpt (n.d.)
[4] MTVG : Multi-text Video Generation with Text-to-Video Models. hf-search. https://huggingface.co/papers/2312.04086 (2023-12-07)
[5] Towards A Better Metric for Text-to-Video Generation. hf-search. https://huggingface.co/papers/2401.07781 (2024-01-15)
[6] STREAM: Spatio-TempoRal Evaluation and Analysis Metric for Video Generative Models. hf-search. https://huggingface.co/papers/2403.09669 (2024-01-30)
[7] WorldJen: An End-to-End Multi-Dimensional Benchmark for Generative Video Models. hf-search. https://huggingface.co/papers/2605.03475 (2026-05-05)
[8] VideoPoet – Google Research. web. https://sites.research.google/videopoet/ (n.d.)
[9] UniVG: Towards UNIfied-modal Video Generation. hf-search. https://huggingface.co/papers/2401.09084 (2024-01-17)
[10] [2609.15863] LynnReal-Omni: Native multi-modal Video Generation for Agentic Visual Workflows. web. https://arxiv.org/abs/2609.15863 (2026-09-14)
[11] TV2TV: A Unified Framework for Interleaved Language and Video Generation. web. https://openaccess.thecvf.com/content/CVPR2026/papers/Han_TV2TV_A_Unified_Framework_for_Interleaved_Language_and_Video_Generation_CVPR_2026_paper.pdf (n.d.)
[12] UnityVideo: Unified Multi-Modal Multi-Task Learning for Enhancing World-Aware Video Generation. web. https://openaccess.thecvf.com/content/CVPR2026/papers/Huang_UnityVideo_Unified_Multi-Modal_Multi-Task_Learning_for_Enhancing_World-Aware_Video_Generation_CVPR_2026_paper.pdf (n.d.)
[13] STREAM: Spatio-TempoRal Evaluation and Analysis Metric for Video Generative Models. web. https://arxiv.org/html/2403.09669v3 (n.d.)
[14] VBench++: Comprehensive and Versatile Benchmark Suite for Video Generative Models. web. https://arxiv.org/html/2411.13503v1 (n.d.)

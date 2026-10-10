# Survey on World Models in AI

## TL;DR

- OuroWorld generates dynamic, looping 3D cinemagraphs from static scenes using vision-language and video models [1].
- World models enable compact policy training in simulated environments by learning compressed representations [2].
- World models allow agents to train in hallucinated dreams and transfer policies to real environments [3].

## Background

World models are generative neural networks that learn compressed representations of environments, enabling agents to train policies within simulated dream environments [2]. These models abstract spatial and temporal information, allowing agents to predict future sensory data based on current motor actions [4]. The concept of world models is inspired by human mental models, which are simplified representations of the real world used for decision-making [4]. Recent advancements have focused on real-world-driven world models for autonomous driving, leveraging diffusion models to generate high-quality driving videos and policies [5]. Additionally, large-scale training of transformer models for long video and language sequences has improved performance on retrieval and understanding tasks [6]. Open-source platforms like CarDreamer provide flexible environments for developing world model-based autonomous driving algorithms [7].

## Foundational Work

World models aim to create generative neural networks that learn compressed representations of environments, enabling compact policy training within simulated dream environments [3]. Early foundational work, such as the 2018 paper, demonstrated how world models can be trained unsupervised to capture spatial and temporal features, allowing agents to learn policies through hallucinated dreams [3]. More recent efforts focus on improving fidelity in agent and object interactions, especially in large-scale ego-centric video datasets, where increasing training data by 100x improves agent modeling but not object fidelity [8]. Additionally, methods like DreamTrue introduce cross-embodiment world models for action-faithful robot simulations, addressing calibration and interaction coverage challenges [9]. While these approaches show promise, limitations such as slow object fidelity improvements and reliance on specific datasets remain [8].

## Recent Developments

Recent developments in world models emphasize dynamic scene generation and improved interaction fidelity. OuroWorld generates photorealistic, explorable 3D scenes with looping motion [1]. A 30,000-hour ego-centric video dataset highlights challenges in modeling agent-object interactions, where agent fidelity improves with data scaling but object fidelity remains limited [8]. DreamTrue addresses these issues by enhancing action following and broadening interaction coverage through geometric calibration and counterfactual post-training [9]. Additionally, OpenWorldLib provides a standardized framework for integrating perception, interaction, and memory in advanced world models [10]. These efforts highlight the growing focus on realism, scalability, and robustness in world model design.

## Evaluation and Limitations

Evaluation of world models focuses on their ability to generate realistic, dynamic environments and their fidelity in simulating interactions. OuroWorld demonstrates dynamic scene generation through 3D cinemagraphs, while the 30,000 hours of ego-centric video highlights the challenges in modeling agent and object interactions [8]. The use of RingAttention in training large models improves retrieval and understanding of long video and language sequences [6]. Limitations include the difficulty in achieving high object fidelity despite increased training data [8], and the need for more robust frameworks to simulate actionable possibilities for reasoning and acting [11]. These evaluations underscore the gap between current capabilities and practical deployment of world models.

## Trends and open problems

World models are advancing toward realistic, interactive, and long-horizon simulations, yet challenges persist. OuroWorld enables dynamic 3D cinemagraphs from static scenes [1], while PAN addresses general, actionable, and long-horizon world simulation, offering improved state representation and training objectives [12]. Wonder supports real-time, camera-controllable exploration, with sparse attention mechanisms enabling efficient memory retrieval [13]. However, object fidelity in human-centric video remains low despite increased data [8], and counterfactual training in robotics faces calibration and coverage issues [9]. Open questions include abstract reasoning, long-term consistency, and architectural design, highlighting a need for unified frameworks like OpenWorldLib [10].

## References
[1] OuroWorld: Bringing Any 3D World Alive as Diverse, Endlessly Looping 3D Cinemagraphs. arxiv. https://arxiv.org/abs/2610.12461 (2026-10-08)
[2] World Models. hf-search. https://huggingface.co/papers/1803.10122 (2018-03-27)
[3] World Models. web. https://arxiv.org/abs/1803.10122 (n.d.)
[4] World Models. web. https://worldmodels.github.io/ (2018-03-27)
[5] DriveDreamer: Towards Real-world-driven World Models for Autonomous Driving. hf-search. https://huggingface.co/papers/2309.09777 (2023-09-18)
[6] World Model on Million-Length Video And Language With RingAttention. hf-search. https://huggingface.co/papers/2402.08268 (2024-02-13)
[7] CarDreamer: Open-Source Learning Platform for World Model based Autonomous Driving. hf-search. https://huggingface.co/papers/2405.09111 (2024-05-15)
[8] What 30,000 Hours of Ego-centric Video Does Not Teach. arxiv. https://arxiv.org/abs/2610.12464 (2026-10-08)
[9] DreamTrue: Action-Faithful Robot World Model with Counterfactual Post-Training. arxiv. https://arxiv.org/abs/2610.12468 (2026-10-08)
[10] OpenWorldLib: A Unified Codebase and Definition of Advanced World Models. hf-search. https://huggingface.co/papers/2604.04707 (2026-04-06)
[11] Critiques of World Models. hf-search. https://huggingface.co/papers/2507.05169 (2025-07-07)
[12] PAN: A World Model for General, Actionable, and Long-Horizon World Simulation | arXiv Science. web. https://arxiv.science/abs/2511.09057 (2026-09-05)
[13] Wonder: Video World Model Done Better. web. https://arxiv.org/abs/2607.26037 (2026-06-17)

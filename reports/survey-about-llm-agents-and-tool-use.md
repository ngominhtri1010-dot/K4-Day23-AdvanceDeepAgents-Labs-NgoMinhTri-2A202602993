# Survey on LLM Agents and Tool Use

## TL;DR

- Recent work (2024 to 2026) has converged on a strategy for defending tool-using LLM agents against indirect prompt injection [1].
- A modular taxonomy and debugging framework improve error detection and recovery in large language model agents, enhancing task success [2].
- The primary research objective in this domain has transitioned from the correctness of single-point calls to the end-to-end executability and robustness of multi-tool chains in complex environments [3].

## Background

LLM agents combine reasoning and acting to enhance task performance [4]. They use tools to gather information and execute actions, improving decision-making [5]. Recent benchmarks highlight the need for better tool selection and usage awareness [5]. These agents address limitations of standalone LLMs by integrating with external systems and APIs [6]. Trustworthiness and safety are critical, with frameworks like TrustAgent focusing on evaluation and mitigation strategies [7]. The field emphasizes efficient design and evaluation mechanisms to ensure reliable and effective agent behavior [8].

## Foundational Work

Foundational work in LLM agents focuses on integrating language models with task-specific tools and optimization frameworks. Agentic BBO studies explore how LLMs can solve complex optimization problems by combining task semantics with mathematical tools [9]. DataSense-Bench evaluates whether AI agents can select and rank training data effectively, which is crucial for fine-tuning models [10]. OnTrack introduces a real-time monitoring system for LLM agent trajectories, preventing irreversible errors during autonomous decision-making [11]. The TrustAgent framework systematically assesses trustworthiness in LLM-based agents, examining intrinsic and extrinsic trust mechanisms [7]. Research also highlights the role of LLMs as autonomous agents, emphasizing their architectural design, tool integration, and cognitive capabilities like reasoning and planning [12]. Modular debugging frameworks help identify and recover from errors in LLM agent systems [2].

## Recent Developments

Recent developments in LLM agents and tool use have focused on enhancing security and efficiency. Out-of-band defenses, such as those in CaMeL, FIDES, Progent, RTBAS, and FORGE, enforce security through deterministic policies rather than relying on model training to refuse malicious instructions [1]. These systems nearly eliminate attacks on the AgentDojo benchmark. Additionally, tool use is being improved through methods like ToolScope, which reduces redundancy and enhances selection accuracy by merging tools and filtering based on context [13]. The MetaTool benchmark highlights the need for better tool selection and usage awareness across scenarios [5]. ToolGym provides a scalable environment for testing agents in complex, real-world scenarios with unreliable tool states [14]. Finally, ToolOmni introduces an agentic framework for open-world tool use, combining proactive retrieval and grounded execution for improved performance [15].

## Evaluation and Limitations

Evaluation of LLM agents focuses on their ability to perform complex tasks by interacting with tools. For instance, ACEBench evaluates tool usage in multi-turn dialogues [16]. Similarly, MCPAgentBench tests LLM agents in realistic task scenarios through dynamic sandbox environments [17]. Additionally, a comprehensive evaluation mechanism for LLM-based agents on manipulating APIs decomposes tool use capability into seven aspects, offering a thorough schema for assessment [6].

Despite these efforts, limitations persist. The integration of mathematically rigorous tools in agentic BBO remains challenging due to varying task domains and system configurations, complicating result comparisons [9]. Furthermore, the ability of AI agents to reliably select the right data for training is still under scrutiny, highlighting gaps in data sensing capabilities [10]. Overall, the evaluation of LLM agents continues to evolve, emphasizing the need for standardized benchmarks and diverse testing environments.

## Trends and open problems

LLM agents are increasingly being used for complex tasks, but their deployment faces several challenges. One key issue is the lack of standardized benchmarks, as noted in [9], where existing agentic BBO studies use different task domains and system configurations, limiting result comparability. Additionally, ensuring safe and reliable agent behavior is critical; [11] highlights that LLM agents often work autonomously with minimal safeguards, leading to potential safety and cost issues. Another open problem is the integration of external tools, where [13] emphasizes that real-world toolsets often contain redundant tools, reducing selection accuracy. Lastly, while out-of-band defenses have been proposed to counter prompt injection attacks, [1] points out the need for adaptive evaluation of these defenses to ensure their effectiveness in real-world scenarios.

## References
[1] Adaptive Evaluation of Out-of-Band Defenses Against Prompt Injection in LLM Agents. arxiv. https://arxiv.org/abs/2606.26479 (2026-06-25)
[2] Where LLM Agents Fail and How They can Learn From Failures. hf-search. https://huggingface.co/papers/2509.25370 (2025-09-29)
[3] [2603.22862] The Evolution of Tool Use in LLM Agents. web. https://arxiv.org/abs/2603.22862 (n.d.)
[4] ReAct: Synergizing Reasoning and Acting in Language Models. web. https://arxiv.org/abs/2210.03629 (2023-03-10)
[5] MetaTool Benchmark for Large Language Models: Deciding Whether to Use Tools and Which to Use. hf-search. https://huggingface.co/papers/2310.03128 (2023-10-04)
[6] An Evaluation Mechanism of LLM-based Agents on Manipulating APIs. web. https://exa.ai/library/publication/nhz281cc2lh (2024-01-01)
[7] A Survey on Trustworthy LLM Agents: Threats and Countermeasures. hf-search. https://huggingface.co/papers/2503.09648 (2025-03-12)
[8] AgentSwift: Efficient LLM Agent Design via Value-guided Hierarchical Search. hf-search. https://huggingface.co/papers/2506.06017 (2025-06-06)
[9] A Closer Look at Agentic BBO: Benchmarking LLM Agents for Black-Box Optimization. arxiv. https://arxiv.org/abs/2610.12183 (2026-10-08)
[10] DataSense-Bench: The First Step Toward an AI Scientist. arxiv. https://arxiv.org/abs/2610.12190 (2026-10-08)
[11] OnTrack: Real-Time Monitoring and Intervention in LLM Agent Trajectories via Streaming Structure-Aware Optimal Transport. arxiv. https://arxiv.org/abs/2610.12375 (2026-10-08)
[12] From Language to Action: A Review of Large Language Models as Autonomous Agents and Tool Users. hf-search. https://huggingface.co/papers/2508.17281 (2025-10-28)
[13] ToolScope: Enhancing LLM Agent Tool Use through Tool Merging and Context-Aware Filtering. hf-search. https://huggingface.co/papers/2510.20036 (2026-05-08)
[14] ToolGym: an Open-world Tool-using Environment for Scalable Agent Testing and Data Curation. hf-search. https://huggingface.co/papers/2601.06328 (2026-01-09)
[15] ToolOmni: Enabling Open-World Tool Use via Agentic learning with Proactive Retrieval and Grounded Execution. web. https://aclanthology.org/2026.acl-long.1736.pdf (n.d.)
[16] ACEBench: Who Wins the Match Point in Tool Usage?. hf-search. https://huggingface.co/papers/2501.12851 (2025-01-22)
[17] MCPAgentBench: A Real-world Task Benchmark for Evaluating LLM Agent MCP Tool Use. hf-search. https://huggingface.co/papers/2512.24565 (2026-01-21)

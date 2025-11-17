# Large Language Models for Optimization


I currently focus on leveraging large language models for optimization, including
- [Surveys](#Surveys)
- [Model Construction](#MO)
- [Algorithm Design](#AD)
- [Solution Verification](#SV)
- [Scene Application](#SA)

 
  
<strong> Last Update: 2025/11/13 </strong>





<a name="Surveys" />

## Surveys 
- [25.01] 组合优化问题的机器学习求解方法, 中国科学：数学 [[Paper](https://kns.cnki.net/kcms2/article/abstract?v=0eC8MkjONMF22O_8ZV904QG42sq6G1NnD79RJp0mZIiDXOkm6HD5KA8AS-kLes3XlqFkt9Hf5i929LNNlS7Mb52hOu3WsmRoOawCQgDETQos3OQpJJPknKu2VDheCI807svvuHH1lSny6qFVbkxWa0kcgGme3wppLp02a9gttJx8Q2elwl9FQQ==&uniplatform=NZKPT&language=CHS)]
- [25.07] Large Language Models for Combinatorial Optimization: A Systematic Review, arXiv [[Paper](https://arxiv.org/abs/2507.03637)]
- [25.08] A Survey of Optimization Modeling Meets LLMs: Progress and Future Directions, arXiv [[Paper](https://arxiv.org/abs/2508.10047)]
- [25.09] Systematic Survey on Large Language Models for Evolutionary Optimization:From Modeling to Solving, arXiv [[Paper](https://arxiv.org/abs/2509.08269)]
- [25.09] Large Language Models and Operations Research: A Structured Survey,, arXiv [[Paper](https://ui.adsabs.harvard.edu/abs/2025arXiv250918180W/abstract)]
- [25.09] Large Language Models and Operations Research: A Structured Survey, arXiv [[Paper]([https://arxiv.org/abs/2509.08269](https://ui.adsabs.harvard.edu/abs/2025arXiv250918180W/abstract))]
- [25.05] A Survey of LLM × DATA, arXiv [[Paper](https://arxiv.org/abs/2505.18458)]  [[Code](https://github.com/weAIDB/awesome-data-llm)]
- [25.03] A Survey on Mathematical Reasoning and Optimization with Large Language Models, arXiv [[Paper](https://arxiv.org/abs/2503.17726)]
- [25.04] Evolutionary Computation in the Era of Large Language Model: Survey and Roadmap, IEEE TEVC [[Paper](https://ieeexplore.ieee.org/document/10767756)] [[Code](https://github.com/wuxingyu-ai/LLM4EC)]
- [24.07] When Large Language Model Meets Optimization, SWarm and Evolutionary Computation [[Paper](https://www.sciencedirect.com/science/article/pii/S2210650224002013)]
- [24.11] Toward Automated Algorithm Design: A Survey and Practical Guide to Meta-Black-Box-Optimization, arXiv [[Paper](https://arxiv.org/abs/2411.00625)]
- [24.10] A Systematic Survey on Large Language Models for Algorithm Design, arXiv [[Paper](https://arxiv.org/abs/2410.14716)]
- [24.01] Artificial Intelligence for Operations Research: Revolutionizing the Operations Research Process, arXiv [[Paper](https://arxiv.org/abs/2401.03244)]
- [24.06] Enhancing Decision-Making in Optimization through LLM-Assisted Inference: A Neural Networks Perspective, IEEE IJCNN [[Paper](https://ieeexplore.ieee.org/abstract/document/10649965)]
- [24.08] Large Language Model-Aided Evolutionary Search for Constrained Multiobjective Optimization, IEEE ICIC [[Paper](https://link.springer.com/chapter/10.1007/978-981-97-5581-3_18)]








<a name="MO" />

## Model Construction
- [23.12] NL4Opt competition: Formulating optimization problems based on their natural language descriptions, NeurIPS [[Paper](https://proceedings.mlr.press/v220/ramamonjison23a/ramamonjison23a.pdf)]
- [22.12] VTCC-NLP at NL4Opt competition subtask 1: An ensemble pre-trained language models for named entity recognition, arXiv [[Paper](https://arxiv.org/abs/2212.07219)]
- [23.01] Opd@ NL4Opt: An ensemble approach for the ner task of the optimization problem, arXiv [[Paper](https://arxiv.org/abs/2301.02459)]
- [22.12] Linear programming word problems formulation using ensemblecrf NER labeler and T5 text generator with data augmentations, arXiv [[Paper](https://arxiv.org/abs/2212.14657)]
- [22.12] Tag embedding and well-defined intermediate representation improve auto-formulation of problem description, arXiv [[Paper](https://arxiv.org/abs/2212.03575)]
- [23.02] A novel approach for auto-formulation of optimization problems,, arXiv [[Paper](https://arxiv.org/abs/2302.04643)]
- [23.08] Highlighting named entities in input for auto-formulation of optimization problems, CICM [[Paper](https://link.springer.com/chapter/10.1007/978-3-031-42753-4_9)]
- [23.11] Dagnosing infeasible optimization problems using large language models, INFOR [[Paper](https://www.tandfonline.com/doi/abs/10.1080/03155986.2024.2385189)]
- [25.02] EquivaMap: Leveraging LLMs for Automatic Equivalence Checking of Optimization Formulations, arXiv [[Paper](https://arxiv.org/abs/2502.14760)]
- [22.12] Augmenting operations research with auto-formulation of optimization models from problem descriptions, EMNLP [[Paper](https://aclanthology.org/2022.emnlp-industry.4/)]
- [24.08] LM4OPT: Unveiling the potential of large language models in formulating mathematical optimization problems, INFOR [[Paper](https://www.tandfonline.com/doi/abs/10.1080/03155986.2024.2388452)]
- [24.10] Towards foundation models for mixed integer linear programming, arXiv [[Paper](https://arxiv.org/abs/2410.08288)]
- [23.08] Holy Grail 2.0: From natural language to constraint models, arXiv [[Paper](https://arxiv.org/abs/2308.01589)]
- [24.07] “I Want It That Way”: Enabling Interactive Decision Support Using Large Language Models and Constraint Programming, TIIS [[Paper](https://dl.acm.org/doi/abs/10.1145/3685053)]
- [22.11] Chain-of-Thought Prompting Elicits Reasoning in Large Language Models, NeurIPS [[Paper](https://proceedings.neurips.cc/paper_files/paper/2022/file/9d5609613524ecf4f15af0f7b31abca4-Paper-Conference.pdf)]
- [23.12] Tree of thoughts: Deliberate problem solving with large language models, NeurIPS [[Paper](https://proceedings.neurips.cc/paper_files/paper/2022/file/9d5609613524ecf4f15af0f7b31abca4-Paper-Conference.pdf)]
- [24.02] Graph of Thoughts: Solving Elaborate Problems with Large Language Models, AAAI [[Paper](https://ojs.aaai.org/index.php/AAAI/article/view/29720)]
- [24.06] Progressive-Hint Prompting Improves Reasoning in Large Language Models, ICML [[Paper](https://openreview.net/forum?id=UkFEs3ciz8&noteId=UkFEs3ciz8)]
- [23.02] ReAct: Synergizing Reasoning and Acting in Language Models, ICLR [[Paper](https://openreview.net/forum?id=WE_vluYUL-X)]
- [24.07] Leveraging Large Language Models for Solving Rare MIP Challenges, arXiv [[Paper](https://arxiv.org/abs/2409.04464)]
- [24.02] Chain-of-Experts: When LLMs Meet Complex Operations Research Problems, ICLR [[Paper](https://openreview.net/forum?id=HobyL1B9CZ)]
- [24.10] Planning Anything with Rigor: General-Purpose Zero-Shot Planning with LLM-based Formalized Programming, arXiv [[Paper](https://arxiv.org/abs/2410.12112)]
- [24.07] OptiMUS-0.3: Using Large Language Models to Model and Solve Optimization Problems at Scale, arXiv [[Paper](https://arxiv.org/abs/2407.19633)]
- [24.08] Optimization modeling and verification from problem specifications using a multi-agent multi-stage LLM framework, INFOR [[Paper](https://www.tandfonline.com/doi/abs/10.1080/03155986.2024.2381306)]
- [25.03] OR-LLM-Agent: Automating Modeling and Solving of Operations Research Optimization Problems with Reasoning LLM, arXiv [[Paper](https://arxiv.org/abs/2503.10009)]
- [25.04] OptimAI: Optimization from natural language using LLM-powered ai agents, arXiv [[Paper](https://arxiv.org/abs/2504.16918)]
- [24.11] Large Language Models for Combinatorial Optimization of Design Structure Matrix, arXiv [[Paper](https://arxiv.org/abs/2411.12571)]
- [25.06] LLM for Large-Scale Optimization Model Auto-Formulation: A Lightweight Few-Shot Learning Approach, SSRN [[Paper](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5329027)]
- [25.01] DRoC: Elevating Large Language Models for Complex Vehicle Routing via Decomposed Retrieval of Constraints, ICLR [[Paper](https://openreview.net/forum?id=s9zoyICZ4k)]
- [23.09] Language Models for Business Optimisation with a Real World Case Study in Production Scheduling, arXiv [[Paper](https://arxiv.org/abs/2309.13218)]
- [25.01] OptiBench Meets ReSocratic: Measure and Improve LLMs for Optimization Modeling, arXiv [[Paper](https://arxiv.org/abs/2309.13218)]
- [24.03] LLaMoCo: Instruction Tuning of Large Language Models for Optimization Code Generation, arXiv [[Paper](https://arxiv.org/abs/2403.01131)]
- [25.05] ORLM: A Customizable Framework in Training Large Models for Automated Optimization Modeling, OR [[Paper](https://pubsonline.informs.org/doi/abs/10.1287/opre.2024.1233)]
- [25.01] LLMOPT: Learning to Define and Solve General Optimization Problems from Scratch, ICLR [[Paper](https://openreview.net/forum?id=9OMvtboTJg)]
- [25.07] Auto-Formulating Dynamic Programming Problems with Large Language Models, arXiv [[Paper](https://arxiv.org/abs/2507.11737)]
- [25.05] OptMATH: A Scalable Bidirectional Data Synthesis Framework for Optimization Modeling, arXiv [[Paper](https://openreview.net/forum?id=9P5e6iE4WK)]
- [24.01] Solving General Natural-Language-Description Optimization Problems with Large Language Models, NAACL [[Paper](https://openreview.net/forum?id=9P5e6iE4WK)]
- [25.02] Evo-Step: Evolutionary Generation and Stepwise Validation for Optimizing LLMs in OR,Openreview [[Paper](https://openreview.net/forum?id=aapUBU9U0D)]
- [25.03] Text2Zinc: A Cross-Domain Dataset for Modeling Optimization and Satisfaction Problems in MiniZinc, arXiv [[Paper](https://arxiv.org/abs/2503.10642)]
- [25.04] Evaluating LLM Reasoning in the Operations Research Domain with ORQA, AAAI [[Paper](https://ojs.aaai.org/index.php/AAAI/article/view/34673)]
- [25.06] CP-Bench: Evaluating Large Language Models for Constraint Modelling, arXiv [[Paper](https://arxiv.org/abs/2506.06052)]




<a name="AD" />

## Algorithm Design
- [25.05] RedAHD: Reduction-Based End-to-End Automatic Heuristic Design with Large Language Models, arXiv [[Paper](https://arxiv.org/abs/2505.20242)]
- [25.03] Combinatorial Optimization for All: Using LLMs to Aid Non-Experts in Improving Optimization Algorithms, arXiv [[Paper](https://arxiv.org/abs/2503.10968)]
- [25.04] Fitness Landscape of Large Language Model-Assisted Automated Algorithm Search, arXiv [[Paper](https://arxiv.org/abs/2504.19636)]
- [25.04] Evolve Cost-Aware Acquisition Functions Using Large Language Models, PPSN [[Paper](https://doi.org/10.1007/978-3-031-70068-2_23)]
- [25.03] PAIR: A Novel Large Language Model-Guided Selection Strategy for Evolutionary Algorithms, PPSN [[Paper](https://arxiv.org/abs/2503.03239)]
- [23.11] Evolution Through Large Models, Springer Nature Link [[Paper](https://link.springer.com/chapter/10.1007/978-981-97-5581-3_18)]
- [24.10] Deep Insights into Automated Optimization with Large Language Models and Evolutionary Algorithms, arXiv [[Paper](https://arxiv.org/abs/2410.20848)]
- [25.05] Generalizable Heuristic Generation Through Large Language Models with Meta-Optimization, arXiv [[Paper](https://arxiv.org/abs/2505.20881)]
- [25.02] Improving Existing Optimization Algorithms with LLMs, arXiv [[Paper](https://arxiv.org/abs/2502.08298)]
- [25.06] EALG: Evolutionary Adversarial Generation of Language Model-Guided Generators for Combinatorial Optimization, arXiv [[Paper](https://arxiv.org/abs/2506.02594)]
- [24.08] Large language model-enhanced algorithm selection: towards comprehensive algorithm representation, ACM IJCAI [[Paper](https://dl.acm.org/doi/abs/10.24963/ijcai.2024/579)]
- [23.12] Mathematical discoveries from program search with large language models, Nature [[Paper](https://www.nature.com/articles/s41586-023-06924-6)]
- [24.12] LLM4AD: A Platform for Algorithm Design with Large Language Model, arXiv [[Paper](https://arxiv.org/abs/2412.17287)]
- [23.11] Algorithm Evolution Using Large Language Model, arXiv [[Paper](https://arxiv.org/abs/2311.15249)]
- [25.04] In-the-loop Hyper-Parameter Optimization for LLM-Based Automated Design of Heuristics, ACM TELO [[Paper](https://arxiv.org/abs/2311.15249)]
- [25.04] Multi-Objective Evolution of Heuristic Using Large Language Model, AAAI [[Paper](https://ojs.aaai.org/index.php/AAAI/article/view/34922)]
- [24.07] Evolution of heuristics: towards efficient automatic algorithm design using large language model, ICML [[Paper](https://dl.acm.org/doi/abs/10.5555/3692070.3693374)]
- [24.12] ReEvo: Large Language Models as Hyper-Heuristics with Reflective Evolution, NeruIPS [[Paper](https://proceedings.neurips.cc/paper_files/paper/2024/file/4ced59d480e07d290b6f29fc8798f195-Paper-Conference.pdf)]
- [24.12] QUBE: Enhancing Automatic Heuristic Design via Quality-Uncertainty Balanced Evolution, arXiv [[Paper](https://arxiv.org/abs/2412.20694)]
- [24.02] AutoSAT: Automatically Optimize SAT Solvers via Large Language Models, arXiv [[Paper](https://arxiv.org/abs/2402.10705)]
- [25.04] HSEvo: Elevating Automatic Heuristic Design with Diversity-Driven Harmony Search and Genetic Algorithm Using LLMs, AAAI [[Paper](https://ojs.aaai.org/index.php/AAAI/article/view/34898)]
- [25.05] Large Language Model-driven Large Neighborhood Search for Large-Scale MILP Problems, ICML [[Paper](https://openreview.net/forum?id=teUg2pMrF0)]
- [24.05] The Importance of Directional Feedback for LLM-based Optimizers, arXiv [[Paper](https://arxiv.org/abs/2405.16434)]
- [25.06] STRCMP: Integrating Graph Structural Priors with Language Models for Combinatorial Optimization, arXiv [[Paper](https://arxiv.org/abs/2506.11057)]
- [24.07] GraphArena: Evaluating and Exploring Large Language Models on Graph Computation, arXiv [[Paper](https://arxiv.org/abs/2407.00379)]
- [24.05] Exploring Combinatorial Problem Solving with Large Language Models: A Case Study on the Travelling Salesman Problem Using GPT-3.5 Turbo, arXiv [[Paper](https://arxiv.org/abs/2405.01997)]
- [25.02] GraphThought: Graph Combinatorial Optimization with Thought Generation, arXiv [[Paper](https://arxiv.org/abs/2502.11607)]
- [24.10] GCoder: Improving Large Language Model for Generalized Graph Problem Solving, arXiv [[Paper](https://arxiv.org/abs/2410.19084)]
- [25.04] Algorithm Discovery With LLMs: Evolutionary Search Meets Reinforcement Learning, arXiv [[Paper](https://arxiv.org/abs/2504.05108)]
- [25.05] CALM: Co-evolution of Algorithms and Language Model for Automatic Heuristic Design, arXiv [[Paper](https://arxiv.org/abs/2505.12285)]
- [24.11] Evolutionary Computation in the Era of Large Language Model: Survey and Roadmap, IEEE TEC [[Paper](https://ieeexplore.ieee.org/abstract/document/10767756)]
- [24.05] LLM as a Complementary Optimizer to Gradient Descent: A Case Study in Prompt Tuning, arXiv [[Paper](https://arxiv.org/abs/2405.19732)]
- [24.03] Identify Critical Nodes in Complex Network with Large Language Models, arXiv [[Paper](https://arxiv.org/abs/2403.03962)]
- [24.09] TS-EoH: An Edge Server Task Scheduling Algorithm Based on Evolution of Heuristic, arXiv [[Paper](https://arxiv.org/abs/2409.09063)]
- [24.10] AutoRNet: Automatically Optimizing Heuristics for Robust Network Design via Large Language Models, arXiv [[Paper](https://arxiv.org/abs/2410.17656)]
- [25.06] ALE-Bench: A Benchmark for Long-Horizon Objective-Driven Algorithm Engineering, arXiv [[Paper](https://arxiv.org/abs/2506.09050)]
- [25.02] ARS: Automatic Routing Solver with Large Language Models, arXiv [[Paper](https://arxiv.org/abs/2502.15359)]
- [24.07] Understanding the Importance of Evolutionary Search in Automated Heuristic Design with Large Language Models, arXiv [[Paper](https://link.springer.com/chapter/10.1007/978-3-031-70068-2_12)]
- [25.04] CO-Bench: Benchmarking Language Model Agents in Algorithm Search for Combinatorial Optimization, arXiv [[Paper](https://arxiv.org/abs/2504.04310)]
- [25.05] A Comprehensive Evaluation of Contemporary ML-Based Solvers for Combinatorial Optimization, arXiv [[Paper](https://arxiv.org/abs/2505.16952)]
- [23.12] Can Language Models Solve Graph Problems in Natural Language, arXiv [[Paper](https://proceedings.neurips.cc/paper_files/paper/2023/file/622afc4edf2824a1b6aaf5afe153fa93-Paper-Conference.pdf)]
- [25.06] OPT-BENCH: Evaluating LLM Agent on Large-Scale Search Spaces Optimization Problems, NeruIPS [[Paper](https://arxiv.org/abs/2506.10764)]
- [25.04] LLaMEA: A Large Language Model Evolutionary Algorithm for Automatically Generating Metaheuristics, IEEE TEVC [[Paper](https://arxiv.org/abs/2506.10764)]
- [24.08] Large Language Models As Evolution Strategies, ACM GECCO [[Paper](https://dl.acm.org/doi/abs/10.1145/3638530.3654238)]
- [25.03] Large language model-based evolutionary optimizer: Reasoning with elitism, Elsevier IJON [[Paper](https://www.sciencedirect.com/science/article/pii/S0925231224020435)]
- [24.07] Leveraging large language model to generate a novel metaheuristic algorithm with CRISPE framework, ACM CLUSTER COMPUT [[Paper](https://www.sciencedirect.com/science/article/pii/S0925231224020435)]
- [25.03] Large Language Model for Multiobjective Evolutionary Optimization, ACM EMO [[Paper](https://dl.acm.org/doi/abs/10.1007/978-981-96-3538-2_13)]
- [24.08] Large Language Model-Aided Evolutionary Search for Constrained Multiobjective Optimization, ACM ICIC [[Paper](https://dl.acm.org/doi/abs/10.1007/978-981-97-5581-3_18)]
- [24.12] Large language models as surrogate models in evolutionary algorithms: A preliminary study, Elsevier SWARM EVOL COMPUT [[Paper](https://www.sciencedirect.com/science/article/pii/S2210650224002797)]
- [25.06] LLMs for Cold-Start Cutting Plane Separator Configuration, Spring Nature Link [[Paper](https://link.springer.com/chapter/10.1007/978-3-031-95976-9_4)]
- [25.06] SMAC3: A Versatile Bayesian Optimization Package for Hyperparameter Optimization, JMLR [[Paper](https://www.jmlr.org/papers/v23/21-0888.html)]
- [24.07] On Different Methods For Automated MILP Solver Configuration, IEEE OPCS [[Paper](https://ieeexplore.ieee.org/abstract/document/10720437)]
- [24.10] Large Language Models as Tuning Agents of Metaheuristics, ESANN [[Paper](https://www.esann.org/sites/default/files/proceedings/2024/ES2024-209.pdf)]







<a name="SV" />

## Solution Verification


<a name="SA" />

## Scene Application



## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=xianchaoxiu/Large-Language-Models-for-Optimization&type=Date)](https://star-history.com/#xianchaoxiu/Large-Language-Models-for-Optimization)

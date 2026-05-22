# Activation Space Interventions Can Be Transferred Between Large Language Models

Source: https://openreview.net/forum?id=HXOicJsmMQ
Source type: openreview-page

---

## Activation Space Interventions Can Be Transferred Between Large Language Models

[![Download PDF](/images/pdf_icon_blue.svg)](/pdf?id=HXOicJsmMQ "Download PDF")

### [Narmeen Fatimah Oozeer](/profile?id=~Narmeen_Fatimah_Oozeer1 "~Narmeen_Fatimah_Oozeer1"), [Dhruv Nathawani](/profile?id=~Dhruv_Nathawani1 "~Dhruv_Nathawani1"), [Nirmalendu Prakash](/profile?id=~Nirmalendu_Prakash2 "~Nirmalendu_Prakash2"), [Michael Lan](/profile?id=~Michael_Lan1 "~Michael_Lan1"), [Abir HARRASSE](/profile?id=~Abir_HARRASSE1 "~Abir_HARRASSE1"), [Amir Abdullah](/profile?id=~Amir_Abdullah1 "~Amir_Abdullah1")

Published: 01 May 2025, Last Modified: 23 Jul 2025ICML 2025 posterEveryone[Revisions](/revisions?id=HXOicJsmMQ)[BibTeX](#)[CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/ "Licensed under Creative Commons Attribution-ShareAlike 4.0 International")

**TL;DR:** We leverage shared activation spaces between models to transfer safety interventions such as steering vectors.

**Abstract:** The study of representation universality in AI models reveals growing convergence across domains, modalities, and architectures. However, the practical applications of representation universality remain largely unexplored. We bridge this gap by demonstrating that safety interventions can be transferred between models through learned mappings of their shared activation spaces. We demonstrate this approach on two well-established AI safety tasks: backdoor removal and refusal of harmful prompts, showing successful transfer of steering vectors that alter the models' outputs in a predictable way. Additionally, we propose a new task, corrupted capabilities, where models are fine-tuned to embed knowledge tied to a backdoor. This tests their ability to separate useful skills from backdoors, reflecting real-world challenges. Extensive experiments across Llama, Qwen and Gemma model families show that our method enables using smaller models to efficiently align larger ones. Furthermore, we demonstrate that autoencoder mappings between base and fine-tuned models can serve as reliable "lightweight safety switches", allowing dynamic toggling between model behaviors.

**Lay Summary:** Recent work has found that large language models can develop similar internal thought patterns to one another, even at different sizes. We design a lightweight translator between language model thought patterns that transfers behaviors from one model to another.
This translator has multiple applications, particularly in AI safety, a field that aims to fix harmful behaviors in models, such as writing harmful code or responding to dangerous requests. Fixing these problems often means retraining the whole model, which is a costly and time consuming process. Our research advances AI safety as it allows us to transfer safe model behaviors from small models to larger models without retraining either model.
For example, if we find a safety switch inside the thoughts of a small model that allows us to change it from being dangerous to harmless, then we can pass this safety switch through our translator to obtain a safety switch in a larger model. This avoids needing to spend resources to find the safety switch in a large model. Our experiments across many models confirm that these translations successfully act as switches that toggle new behaviors on demand.

**Link To Code:** https://github.com/withmartian/Closing-Backdoors-Via-Representation-Transfer

**Primary Area:** Social Aspects->Accountability, Transparency, and Interpretability

**Keywords:** Representation Transfer, AI Safety, Mechanistic Interpretability, Refusal Vector, Sleeper Agents, Backdoors

**Submission Number:** 9214

Loading

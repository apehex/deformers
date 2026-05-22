# UnifyVocab

Source: https://openreview.net/forum?id=CP6CAqxAGJ
Source type: openreview-page

---

## Unifying Vocabulary of Large Language Model with Statistical Token-level Alignment

[![Download PDF](/images/pdf_icon_blue.svg)](/pdf?id=CP6CAqxAGJ "Download PDF")

### [Chong Li](/profile?id=~Chong_Li6 "~Chong_Li6"), [Jiajun Zhang](/profile?id=~Jiajun_Zhang1 "~Jiajun_Zhang1"), [Chengqing Zong](/profile?id=~Chengqing_Zong1 "~Chengqing_Zong1")

19 Sept 2024 (modified: 05 Feb 2025)Submitted to ICLR 2025Everyone[Revisions](/revisions?id=CP6CAqxAGJ)[BibTeX](#)[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/ "Licensed under Creative Commons Attribution 4.0 International")

**Keywords:** Vocabulary Adaptation, Large Language Model, Efficient NLP

**TL;DR:** We propose a method to align and replace the vocabulary of large language models using only 10B tokens, and find it facilities deep knowledge transfer between models like token-level distillation.

**Abstract:** Large Language Models (LLMs) achieve great success across many general tasks, but the mismatch among different vocabularies hinders further applications like token-level distillation and inference with various models. To align the vocabularies of LLMs, we propose a simple yet effective method named \*\*UnifyVocab\*\* to replace the vocabulary of an LLM at a limited cost. A new vocabulary alignment method is devised first to align the source vocabulary to the target one. We then rearrange the corresponding parameters like embeddings, and progressively fine-tune the model. Experimental results on models across multiple parameter scales demonstrate the effectiveness and generalization of UnifyVocab, which costs as few as 10B tokens to recover 98.02\% performance of the vanilla models on average. We further find that unifying the vocabularies significantly facilitates the token-level distillation which remarkably boosts (+4.4\%) the model with only 235M tokens. Moreover, our method provides a better initialization of multilingual vocabulary for LLMs to adapt to new languages.

**Primary Area:** generative models

**Code Of Ethics:** I acknowledge that I and all co-authors of this work have read and commit to adhering to the ICLR Code of Ethics.

**Submission Guidelines:** I certify that this submission complies with the submission instructions as described on https://iclr.cc/Conferences/2025/AuthorGuide.

**Anonymous Url:** I certify that there is no URL (e.g., github page) that could be used to find authors’ identity.

**No Acknowledgement Section:** I certify that there is no acknowledgement section in this submission for double blind review.

**Submission Number:** 1726

Loading

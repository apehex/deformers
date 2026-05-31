---
title: "Don't Trust Yourself"
source: "https://www.kaggle.com/competitions/openai-gpt-oss-20b-red-teaming/writeups/dont-trust-the-yes-man"
---

I reviewed Reddit discussion of the GPT-5 rollout. While users noticed that their tone changed, various reports of connections with "spirits", etc. seemed to persist. That made me curious if deliberative alignment and other post-training for gpt-oss was resilient to conversation histories that were already bad.

I decided to test this methodically.

In this project, I

1.   Developed an example policy document that attempted to categorize all publicly known (and stumbled upon) policy lines for the model
2.   Developed a series of scenarios designed to exercise each policy category
3.   Generated sycophantic conversation histories across a variety of turn counts with a looser model (Grok-4 in this case).
4.   Tested if gpt-oss-20b would generate violating content relative to my policy for each (policy category, turn count) pair

After ~800 conversations, to my surprise it seems that performance was uneven. Violation rate increases to 20% by turn 20 but then for some reason the model gets its head back at around 30 turns but then loses the plot again at 50 turns.

![Image 1](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F352465%2F4d8f7017884ad43f85ba0a8bdb8674df%2Foutput2.png?generation=1756242522584443&alt=media)

Digging into individual categories the model struggles most with adult sexual content, legal and investment matters. When pushed, the model can also be tricked into glorifying self-harm for the purpose of environmentalism, but this didn't show up in the broader sweep.

![Image 2](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F352465%2Fb383e935c53de6f6863d7093a95d7121%2Foutput.png?generation=1756242270942327&alt=media)

I also tested if the presumed gender of the user would have an effect. The model was more violating with a presumed female user, but also more refusing. This may indicate a gender bias the new safe-completion training data.

![Image 3](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F352465%2Fb5e00b46f97da8f1808c48f0ac9c0d30%2Foutput3.png?generation=1756242529453680&alt=media)

In conclusion, further work is needed to shore up longer context safety (in turn count, not necessarily token count as all these convos were <2k tokens). While it's encouraging to see many of the worst categories well defended, users of gpt-oss-20b will be exposed to the minimum safety available at any point in their conversation history, as gpt-oss-20b defers to "the previous assistant" when in doubt.

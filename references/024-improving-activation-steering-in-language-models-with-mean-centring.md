# Improving Activation Steering in Language Models with Mean-Centring

Source: https://arxiv.org/abs/2312.03813
Fetched URL: https://arxiv.org/pdf/2312.03813
Source type: arxiv-pdf

---

## Page 1

Improving Activation Steering in Language Models with Mean-Centring
Ole Jorgensen1*
Dylan Cope1,2
Nandi Schoots1,2
Murray Shanahan1
1Imperial College London
2King’s College London
Abstract
Recent work in activation steering has demonstrated the po-
tential to better control the outputs of Large Language Models
(LLMs), but it involves finding steering vectors. This is dif-
ficult because engineers do not typically know how features
are represented in these models. We seek to address this issue
by applying the idea of mean-centring to steering vectors. We
find that taking the average of activations associated with a
target dataset, and then subtracting the mean of all training
activations, results in effective steering vectors. We test this
method on a variety of models on natural language tasks by
steering away from generating toxic text, and steering the
completion of a story towards a target genre. We also apply
mean-centring to extract function vectors, more effectively
triggering the execution of a range of natural language tasks
by a significant margin (compared to previous baselines). This
suggests that mean-centring can be used to easily improve the
effectiveness of activation steering in a wide range of contexts.
1
Introduction
Large Language Models (LLMs) have become increasingly
capable over the past few years across a diverse range of
tasks (Peters et al. 2018; Radford et al. 2019; OpenAI 2023).
However, in part due to a lack of understanding of how these
capabilities are implemented, we are unable to address issues
such as social biases (Abid, Farooqi, and Zou 2021). Some
approaches to mitigating these issues modify the weights
of the LLM (Ilharco et al. 2023; Meng et al. 2022), but
these techniques either require fine-tuning or have only been
applied to editing factual associations encoded in the model.
A recent approach to controlling LLMs is activation steer-
ing (Turner et al. 2023; Li et al. 2023; Subramani, Suresh, and
Peters 2022), or similarly representation engineering (Zou
et al. 2023). Activation steering aims to extract features from
language models to better control their outputs. It typically
does this by making inference-time modifications to some
activations of the model.
In this work, we apply activation steering to incorporate
some behaviour exhibited by an arbitrary dataset D into the
output of a language model. This introduces a simple pipeline
for modifying language model behaviour, which current ac-
tivation steering methods do not allow for in full generality.
They either require the identification of an opposite behaviour
*ojorgensen1417@gmail.com
(Counterbalanced Subtractions in Turner et al. (2023)), suc-
cinctly describing the pertinent behaviour of the dataset (LAT
Scans in Zou et al. (2023)), or are computationally expensive
(training a sparse autoencoder on language model activations
(Cunningham et al. 2023; Bricken et al. 2023)).
Our paper aims to address these issues by applying a sim-
ple processing technique to steering vectors, in the spirit of
similar work in word representations (Mu and Viswanath
2018). Our technique, which we call mean-centring, success-
fully incorporates properties of datasets into the outputs of
LLMs, whilst maintaining coherence. This provides a simple
method for changing model behaviour using only a dataset,
making it easier to apply activation steering in a wider range
of contexts. In summary:
• In Section 3 we introduce mean-centring as a method for
creating better steering vectors.
• In Section 4.1 we demonstrate the efficacy of mean-
centring by controlling a language model to generate
non-toxic continuations of toxic comments.
• In Section 4.2 we show that mean-centring increases the
range of tasks for which steering can be applied as com-
pared to methods that require a counterbalancing concept.
We demonstrate the efficacy of mean-centring by influ-
encing the genres of stories as they are generated.
• In Section 4.3 we demonstrate the efficacy of mean-
centring by extracting more effective function vectors,
compared to non mean-centred approaches. This leads
to significant improvements in accuracy over previous
baselines.
2
Related Work
2.1
Linear Representation Hypothesis
The linear representation hypothesis (Elhage et al. 2022)
proposes that many human-interpretable high-level concepts
are represented linearly as directions in the residual stream of
language models. There is significant evidence for the linear
structure of neural network representations, including linear
operations on Word2Vec embeddings capturing semantic
meaning (Mikolov, Yih, and Zweig 2013). There is strong
evidence in the context of language models specifically, due
to the success of linear probes and edits locating information
arXiv:2312.03813v1  [cs.CL]  6 Dec 2023

## Page 2

Fantasy
Sci-fi
Sports
enchanted
humankind
exhilar
mystical
humanity
victorious
magical
mankind
triumph
awakened
millennia
cheering
sorce
interstellar
triumphant
(a) Using Mean-Centring
Fantasy
Sci-fi
Sports
unthinkable
unthinkable
unthinkable
enormous
enormous
enormous
massive
massive
massive
immense
immense
immense
fateful
fateful
fateful
(b) Not Mean-Centring
Table 1: Using datasets of stories with different genres (Section 4.2) we extract vectors with and without mean-centring (f and
µtarget in Figure 1) at layer 29 of GPT-2 XL. These tables show the top-5 tokens ranked by the inner product between the token
and extracted vector, as developed by nostalgebrist (2020). Mean-centring greatly improves the relevance of the tokens to the
genre, demonstrating that the method finds distillation vectors that effectively capture the key concept for a target dataset.
Figure 1: An example of mean-centring illustrated on a set
of highly anisotropic activations (i.e. offset from the origin).
When steering, we want to use the vector f which generates
some target behaviour. We compute this by averaging acti-
vations from a dataset exhibiting this behaviour, µtarget and
subtracting the mean across all training examples b.
within models (Meng et al. 2022; Nanda, Lee, and Wattenberg
2023; Gurnee and Tegmark 2023).
Recent advances in learning the representations of con-
cepts in language models using sparse auto-encoders provides
substantial further evidence for this hypothesis (Cunningham
et al. 2023; Bricken et al. 2023). This suggests that if we find
the right vector to represent a concept, then we can steer any
residual stream activation into the direction of that concept by
simply adding that vector to the activation (Zou et al. 2023).
2.2
Activation Steering
There have been recent efforts to control the outputs of lan-
guage models through activation steering, i.e. adding vectors
into the activations of a model at inference time. The general
aim of activation steering is to introduce some property into
the output of a model by identifying some steering vector
f and adding it to some layer(s) of the forward pass of a
model, at some token position(s). This has been applied to
incorporating features such as how “loving” a text is (Turner
et al. 2023), functions such as reciting the capital of a country
(Todd et al. 2023), and improving the truthfulness of text (Li
et al. 2023).
Algorithm 1: Mean-Centred Activation Steering
Input:
M = language model
p = user prompt
Dtraining = training dataset sample
Dtarget = target dataset
SteeringMethod = Method used to steer language model
Output:
S = steered output text
1: M.forward(Dtarget)
2: µtarget = Mean(M.activations)
3: M.forward(Dtraining)
4: µtraining = Mean(M.activations)
5: v ←µtarget −µtraining
6: S ←SteeringMethod(M,p,v)
In this paper we will always add a steering vector f to the
final token position, and at a single layer.
2.3
Anisotropy
An alternative way of understanding the activations of lan-
guage models comes from analysing their geometric structure.
Multiple works have demonstrated the anisotropy of the ac-
tivations of language models (Ethayarajh 2019; Cai et al.
2021). Anisotropic activations are not distributed uniformly
around the zero point in activation space, but instead are
offset in a consistent direction. A similar phenomena was
also identified in classical word representations in NLP such
as word2vec (Mikolov, Yih, and Zweig 2013) and GLoVE
(Pennington, Socher, and Manning 2014). Mu and Viswanath
(2018) improve downstream performance on these word rep-
resentations by subtracting the mean, and then projecting on
the dominant remaining directions. This directly inspires our
own method of mean-centring.
3
Mean-Centred Activation Steering
The method that we propose aims to get an LLM to ex-
hibit behaviours that are not well-defined, but that can be
captured by a dataset of examples that demonstrate the be-
haviour. Therefore, in our method we use a target dataset
made of examples of a target behaviour to extract a distilla-
tion vector that can be used to get an LLM to generate the

## Page 3

0
10
20
30
40
50
60
Words Generated After Prompt
0.0
0.2
0.4
0.6
0.8
1.0
Average Sentiment
Mean-centring (Non-Toxic)
Mean-centring (Loving)
No-centring (Loving)
ActAdd (Turner et al.)
Unsteered
(a) Changes in mean positive sentiment of generated text with each
word generated for the different methods (with 95% CI bands).
7.0
7.5
8.0
8.5
9.0
9.5
10.0
Negative Toxicity Log-Probability
Generation Method
Mean-centring (Loving)
ActAdd (Turner et al.)
Mean-centring (Non-Toxic)
No-centring (Loving)
Unsteered
(b) Negative toxicity log-probabilities for the different steering meth-
ods (higher means less toxic), showing toxicity reductions for mean-
centred steering and ActAdd (Turner et al. 2023).
Figure 2: Results from Toxicity Removal Experiments
target behaviour.
Let x1, . . . , xn be the residual stream activations at some
layer l across all token positions of an LLM when performing
inference on a target dataset Dtarget of exemplary behaviour.
From the set of activations x1, . . . , xn we want to extract
a distillation vector f. Previous work (Cai et al. 2021) has
demonstrated that the activations of GPT-2 Small and BERT
activations typically have a non-zero mean (Section 2.3),
across all layers. In Appendix A we replicate these findings
for a range of open source language models. This means that
we might decompose the activations xi as
xi = αif + b + vi,
(1)
where f is the representation of the behaviour displayed in the
dataset Dtarget, b is the bias vector applied to all activations in
the language model, and vi is a noise vector encoding infor-
mation about behaviour not shared by the other datapoints in
Dtarget. The mean of these activations can now be described
µtarget := 1
n
n
X
i=1
αif + 1
n
n
X
i=1
vi + b.
(2)
If we assume that the noise vectors, vi, are distributed inde-
pendently about the 0 ∈Rdmodel vector, then by the law of
large numbers the mean of all activations xi becomes
µtarget →αf + b
as
n →∞.
(3)
Steering with µtarget directly might be sub-optimal, since
the bias vector b in Equation 3 has significant magnitude and
does not encode any information specific to the dataset Dtarget.
We demonstrate its ineffectiveness empirically in Section 4.
Instead, we remove the bias vector b from µtarget. Assum-
ing that averaging activations x′
1, . . . , x′
n′ of samples of the
training distribution, Dtraining, approximates b:
µtraining := 1
n′
n′
X
i=1
x′
i ≈b,
(4)
allows us to extract the vector via f ≈µtarget −µtraining.
We call this method of extracting distillation vectors mean-
centring, which we illustrate in Figure 1. We present the
algorithm used to implement mean-centred activation steer-
ing in Algorithm 1.
4
Experimental Evaluations
In this section we evaluate mean-centring in three different
contexts. We firstly evaluate its effectiveness at removing tox-
icity from language models (Section 4.1), demonstrating that
it is comparable to an existing steering method, namely coun-
terbalanced subtractions from Turner et al. (2023). We then
apply mean-centring to two domains where techniques like
counterbalanced subtractions or LAT Scans are not applica-
ble: steering the genre of stories (Section 4.2) and extracting
better function vectors (Section 4.3).
We perform experiments on GPT-2 Small, Medium, Large
and XL (Radford et al. 2019), GPT-J-6B (Wang and Komat-
suzaki 2021), GPT-NeoX-20B (Black et al. 2022), Llama-2
7B and Llama-2 13B (Touvron et al. 2023). In Appendix C
we give detailed information about the datasets that we use.
4.1
Removing Toxicity Experiments
In this section we demonstrate the efficacy of mean-centring
in reducing the toxicity of language models. We prompt GPT-
2 Small to generate continuations of toxic comments, where
prompts are created using a derivative of the Jigsaw Toxic
Comments dataset (Adams et al. 2017; Borkan et al. 2019)
that only included toxic comments (Appendix C.31). We took
the first half of each comment and used GPT-2 Small to
generate continuations, using each of the following methods
of steering:
• Mean-centring (Non-Toxic): Using mean-centring with a
dataset of ‘non-toxic’ text, a subset of the Jigsaw dataset
filtered to only include non-toxic comments.
• Mean-centring (Loving): Using mean-centring with the
Loving dataset containing ‘loving’ text generated by GPT-
3.5 (Appendix C.5 for details of this dataset).
• No-centring (Loving): Using the average of the activations
associated with the Loving dataset (Appendix C.5), but
without mean-centring.
• ActAdd: Using the ActAdd method from Turner et al.
(2023) with the prompt ‘Love’ counterbalanced by ‘Hate’.
• Unsteered: Standard inference.
1Warning: examples of offensive and hateful comments appear
in the Appendix, but none appear in the main paper contents

## Page 4

scifi
sports
Genre
0.000
0.025
0.050
Average Genre
Word Frequency
Sci-fi to Sports
fantasy
scifi
Genre
0.00
0.02
0.04
0.06
Fantasy to Sci-fi
sports
fantasy
Genre
0.000
0.025
0.050
Sports to Fantasy
Unsteered
Mean-centred
Figure 3: Genre-word frequencies in generated text continuing stories of a given genre. Text generated without steering
(Unsteered) is compared to text generated using mean-centring with a target genre’s dataset (Mean-centred). Mean-centring
consistently reduces the frequency of words in the genre we steer away from, and increases it in the genre we steer towards.
To evaluate the generated text, we use two pretrained mod-
els trained to classify positive sentiment and toxicity sepa-
rately. First, we use a DistilBERT (Sanh et al. 2020) model
with a sentiment head trained on the Stanford Sentiment Tree-
bank (SST-2) dataset (Socher et al. 2013) to compute positive
sentiment values. Second, we use a RoBERTa model (Liu
et al. 2019) trained on the Jigsaw dataset to classify toxic-
ity (Dale et al. 2021). We take the log-probability of being
classified as ‘toxic’ to evaluate the toxicity of a text. We
perform a hyperparameter sweep that minimises toxicity to
fairly compare the methods (see Appendix D.2 for details).
Once the best hyperparameters have been selected, Figure
2 displays the results of the final steering methods. Figure
2a shows the average sentiment and Figure 2b shows the
negative toxicity log-probability of generated text, a higher
value respectively represents more positive sentiment and
lower toxicity. We find that for both average sentiment and
negative toxicity log-probability, the mean-centring (Loving)
method is superior to all other methods we investigate, and in
particular to the no-centring (Loving) method. We also find
that the mean-centring (Non-Toxic) method is able to reduce
the toxicity of the model without substantially increasing the
sentiment of responses. This demonstrates that one can con-
trol mean-centring steering methods effectively by choosing
appropriate datasets.
Appendix E includes examples of steered comment com-
pletions using the different methods.
4.2
Steering Story Continuations
The above experiments demonstrate the comparable effec-
tiveness of mean-centring compared to counterbalanced sub-
tractions. However, a big benefit of mean-centring is that we
can easily apply it to situations where it is not clear how to
use counterbalanced subtractions. One such example is in
changing the genre of stories.
GPT-2 Small was prompted with the beginning of a story
in a fantasy, sci-fi, or sports genre, before mean-centred steer-
ing is used to produce continuations of the story in another
genre. We provide evidence that the mean-centred distillation
vectors are more interpretable than the non mean-centred dis-
tillation vectors in Table 1 and Appendix B using the Logit
Lens, as introduced by (nostalgebrist 2020).
In order to measure the effects of steering, we took each
of the story datasets and found the sets of word stems that
are unique to each dataset and appear at least twice. Then for
any given sample of text, we can compute the frequencies in
which genre-specific word stems appear. In Figure 3 we show
the results for three experiments in which we cut each of the
stories from the different datasets in half and then generated
80 tokens from these prompts, steered with a distillation
vector extracted from a target dataset (with hyperparameters
l = 3, λ = 60). For all three plots, we find that mean-centred
steering towards a genre increases the frequency of words
related to that genre compared to the unsteered model. See
Table 2 for an example with Llama-2 7B, and Appendix F
for examples of steered stories with GPT-2 Small.
Unsteered Continuation
Steered Using Fantasy
Distillation Vector
Yesterday, my son was out
kicking a football. Then
he came in and said, “Mom,
I’m going to be a profes-
sional football player when
I grow up.” “That’s great,”
I said.
Yesterday, my son was out
kicking a football. Then
he came inside and told me
that he had found a strange
creature in the garden. I
rushed outside to see what
it was. It was a magical
fairy!
Table 2: Mean-centred steering applied to Llama-2 7B with
the distillation vector extracted from the fantasy dataset. The
vector is applied at the final token at layer l = 25, and it is
scaled by a factor of 3. Bold indicates input prompt.
4.3
Better Function Vectors
As a final application of mean-centring in a domain where
counterbalanced subtractions cannot be applied, we consider
recent work on extracting function vectors by Todd et al.
(2023). The premise of this work is to extract a vector in
the activations of a language model which corresponds to
an input-output function, such as a function which takes in
a country and returns its capital. Adding this vector should
then cause the model to imitate this function accurately.
For example, when prompting a language model with
“England: ”, steering with a function vector that triggers the
country-capital function, FVcountry-capital, should lead a model

## Page 5

8
10
12
14
16
18
Layer
0.1
0.2
0.3
0.4
0.5
Average Accuracy
Uncentred
Mean-centred
No Steering
(a) Average accuracy across 6 different tasks for each layer.
antonym
capitalize
country-capital english-french
present-past
singular-plural
Task
0.0
0.2
0.4
0.6
0.8
Average Accuracy
No-centring
Mean-centred
(b) Steering in Layer 15 for each task (5 random seeds).
Figure 4: Average accuracy (with 95% CI error bars) plots for steering GPT-J-6B with the uncentred and mean-centred method,
as well as the average accuracy without steering.
to output “London”.
Although the authors present a more complicated method
for producing this function vector, their baseline method
for producing function vectors consists of simply taking the
average of activations associated with in-context learning ex-
amples of the desired behaviour. We can apply mean-centring
to this by simply subtracting the mean of some training acti-
vations for the model.
Figure 4a demonstrates that incorporating mean-centring
for GPT-J-6B (using the same datasets and evaluation method
in the zero-shot context described by (Todd et al. 2023))
improves accuracy in most layers, sometimes substantially.
Using mean-centring at layer 15 gives an accuracy of 45.7%
across the 6 tasks studied, which is significantly better than
the accuracy without mean-centring of 29.2%. Figure 4b
shows that this improvement is due to minor improvements
across the antonym, capitalize, present-past and singular-
plural tasks, and significant improvements in the country-
capital and english-french tasks.
5
Conclusion
Language model activations are typically not centred around
the origin, but are instead offset in some consistent direction.
We develop a new approach for activation steering, mean-
centring, which accounts for this by subtracting the offset.
We demonstrate that mean-centring has two key benefits:
1) it increases performance as compared to no-centring; 2)
the method is versatile and can be applied to a wider range
of domains than counterbalancing methods.
We hypothesize that other methods such as LAT scans
(Zou et al. 2023) and counterbalanced subtractions (Turner
et al. 2023) may implicitly perform mean-centring. By intro-
ducing mean-centring explicitly we are able to easily apply
activation steering to domains in which there is no obvious
concept to counterbalance with. This could allow for other
researchers to easily use activation steering in their own work,
with only a dataset exhibiting the desired behaviour. This may
simplify carrying out many of the safety-relevant applications
of activation steering such as red-teaming (Rimsky 2023) and
narrowing model capabilities (Belrose et al. 2023).
Limitations and Future Work. Although mean-centring
does improve model performance at the best layer for GPT-J,
it does not improve performance at all layers. We hypothesize
that the models for which mean-centring provides the biggest
advantage are those models for which anisotropy is most
pronounced, but Appendix A doesn’t suggest that changes in
anisotropy between layers predicts the performance of mean-
centring. Thus, investigating the link between anisotropy and
improvements in accuracy would be useful here, as well as
investigating other relevant factors which predict the success
of mean-centring.
Cai et al. (2021) present evidence for other structures in
activation geometries, including distinct clustering. Future
work could investigate the extent to which accounting for
these aspects could lead to further improvements to steering.
References
Abid, A.; Farooqi, M.; and Zou, J. 2021. Persistent Anti-
Muslim Bias in Large Language Models. In Proceedings of
the 2021 AAAI/ACM Conference on AI, Ethics, and Society,
AIES ’21, 298–306. New York, NY, USA: Association for
Computing Machinery. ISBN 9781450384735.
Adams, C.; Sorensen, J.; Elliott, J.; Dixon, L.; McDonald,
M.; nithum; and Cukierski, W. 2017. Toxic Comment Classi-
fication Challenge.
Belrose, N.; Schneider-Joseph, D.; Ravfogel, S.; Cotterell,
R.; Raff, E.; and Biderman, S. 2023. LEACE: Perfect linear
concept erasure in closed form. arXiv:2306.03819.
Black, S.; Biderman, S.; Hallahan, E.; Anthony, Q.; Gao,
L.; Golding, L.; He, H.; Leahy, C.; McDonell, K.; Phang,
J.; Pieler, M.; Prashanth, U. S.; Purohit, S.; Reynolds, L.;
Tow, J.; Wang, B.; and Weinbach, S. 2022. GPT-NeoX-20B:
An Open-Source Autoregressive Language Model. In Fan,
A.; Ilic, S.; Wolf, T.; and Gall´e, M., eds., Proceedings of
BigScience Episode #5 – Workshop on Challenges & Per-
spectives in Creating Large Language Models, 95–136. vir-
tual+Dublin: Association for Computational Linguistics.
Borkan, D.; Dixon, L.; Sorensen, J.; Thain, N.; and Vasser-
man, L. 2019. Nuanced Metrics for Measuring Unintended
Bias with Real Data for Text Classification. In Compan-
ion Proceedings of The 2019 World Wide Web Conference,
491–500. Association for Computing Machinery.
Bricken, T.; Templeton, A.; Batson, J.; Chen, B.; Jermyn,
A.; Conerly, T.; Turner, N.; Anil, C.; Denison, C.; Askell,
A.; Lasenby, R.; Wu, Y.; Kravec, S.; Schiefer, N.; Maxwell,

## Page 6

T.; Joseph, N.; Hatfield-Dodds, Z.; Tamkin, A.; Nguyen,
K.; McLean, B.; Burke, J. E.; Hume, T.; Carter, S.;
Henighan, T.; and Olah, C. 2023. Towards Monosemantic-
ity: Decomposing Language Models With Dictionary Learn-
ing.
Transformer Circuits Thread.
Https://transformer-
circuits.pub/2023/monosemantic-features/index.html.
Cai, X.; Huang, J.; Bian, Y.; and Church, K. 2021. Isotropy
in the Contextual Embedding Space: Clusters and Manifolds.
In International Conference on Learning Representations.
Cunningham, H.; Ewart, A.; Riggs, L.; Huben, R.; and
Sharkey, L. 2023. Sparse Autoencoders Find Highly In-
terpretable Features in Language Models. arXiv:2309.08600.
Dale, D.; Markov, I.; Logacheva, V.; Kozlova, O.; Semenov,
N.; and Panchenko, A. 2021. SkoltechNLP at SemEval-2021
Task 5: Leveraging Sentence-level Pre-training for Toxic
Span Detection. In Proceedings of the 15th International
Workshop on Semantic Evaluation (SemEval-2021), 927–934.
Elhage, N.; Hume, T.; Olsson, C.; Schiefer, N.; Henighan,
T.; Kravec, S.; Hatfield-Dodds, Z.; Lasenby, R.; Drain, D.;
Chen, C.; Grosse, R.; McCandlish, S.; Kaplan, J.; Amodei,
D.; Wattenberg, M.; and Olah, C. 2022. Toy Models of
Superposition. Transformer Circuits Thread.
Ethayarajh, K. 2019. How Contextual are Contextualized
Word Representations? Comparing the Geometry of BERT,
ELMo, and GPT-2 Embeddings. In Proceedings of the 2019
Conference on Empirical Methods in Natural Language Pro-
cessing and the 9th International Joint Conference on Natu-
ral Language Processing (EMNLP-IJCNLP), 55–65. Hong
Kong, China: Association for Computational Linguistics.
Gokaslan, A.; and Cohen, V. 2019. OpenWebText Corpus.
Gurnee, W.; and Tegmark, M. 2023. Language Models Rep-
resent Space and Time. arXiv:2310.02207.
Ilharco, G.; Ribeiro, M. T.; Wortsman, M.; Schmidt, L.; Ha-
jishirzi, H.; and Farhadi, A. 2023. Editing models with task
arithmetic. In The Eleventh International Conference on
Learning Representations.
Li, K.; Patel, O.; Vi´egas, F.; Pfister, H.; and Wattenberg,
M. 2023. Inference-Time Intervention: Eliciting Truthful
Answers from a Language Model. In Advances in Neural
Information Processing Systems.
Liu, Y.; Ott, M.; Goyal, N.; Du, J.; Joshi, M.; Chen, D.;
Levy, O.; Lewis, M.; Zettlemoyer, L.; and Stoyanov, V. 2019.
RoBERTa: A Robustly Optimized BERT Pretraining Ap-
proach. arXiv:1907.11692.
Meng, K.; Bau, D.; Andonian, A. J.; and Belinkov, Y. 2022.
Locating and Editing Factual Associations in GPT. In Ad-
vances in Neural Information Processing Systems.
Mikolov, T.; Yih, W.-t.; and Zweig, G. 2013. Linguistic
Regularities in Continuous Space Word Representations. In
Proceedings of the 2013 Conference of the North American
Chapter of the Association for Computational Linguistics:
Human Language Technologies, 746–751.
Mu, J.; and Viswanath, P. 2018. All-but-the-Top: Simple
and Effective Postprocessing for Word Representations. In
6th International Conference on Learning Representations,
ICLR 2018, Vancouver, BC, Canada, April 30 - May 3, 2018,
Conference Track Proceedings.
Nanda, N.; Lee, A.; and Wattenberg, M. 2023. Emergent
Linear Representations in World Models of Self-Supervised
Sequence Models. arXiv:2309.00941.
nostalgebrist. 2020. Interpreting GPT: The Logit Lens.
OpenAI. 2023. GPT-4 Technical Report. arXiv:2303.08774.
Pennington, J.; Socher, R.; and Manning, C. 2014. GloVe:
Global Vectors for Word Representation. In Moschitti, A.;
Pang, B.; and Daelemans, W., eds., Proceedings of the 2014
Conference on Empirical Methods in Natural Language Pro-
cessing (EMNLP), 1532–1543. Doha, Qatar: Association for
Computational Linguistics.
Peters, M. E.; Neumann, M.; Iyyer, M.; Gardner, M.; Clark,
C.; Lee, K.; and Zettlemoyer, L. 2018. Deep contextualized
word representations. arXiv:1802.05365.
Radford, A.; Wu, J.; Child, R.; Luan, D.; Amodei, D.; and
Sutskever, I. 2019. Language Models are Unsupervised Mul-
titask Learners. Technical report, OpenAI.
Rimsky, N. 2023. Red-teaming language models via activa-
tion engineering.
Sanh, V.; Debut, L.; Chaumond, J.; and Wolf, T. 2020. Distil-
BERT, a distilled version of BERT: smaller, faster, cheaper
and lighter. arXiv:1910.01108.
Socher, R.; Perelygin, A.; Wu, J.; Chuang, J.; Manning, C. D.;
Ng, A.; and Potts, C. 2013. Recursive Deep Models for Se-
mantic Compositionality Over a Sentiment Treebank. In
Proceedings of the 2013 Conference on Empirical Methods
in Natural Language Processing (EMNLP), 1631–1642. As-
sociation for Computational Linguistics.
Subramani, N.; Suresh, N.; and Peters, M. 2022. Extracting
Latent Steering Vectors from Pretrained Language Models.
In Findings of the Association for Computational Linguistics:
ACL 2022.
Todd, E.; Li, M. L.; Sharma, A. S.; Mueller, A.; Wallace,
B. C.; and Bau, D. 2023. Function Vectors in Large Language
Models. arXiv:2310.15213.
Touvron, H.; Martin, L.; Stone, K.; Albert, P.; Almahairi, A.;
Babaei, Y.; Bashlykov, N.; Batra, S.; Bhargava, P.; Bhosale,
S.; Bikel, D.; Blecher, L.; Ferrer, C. C.; Chen, M.; Cucurull,
G.; Esiobu, D.; Fernandes, J.; Fu, J.; Fu, W.; Fuller, B.; Gao,
C.; Goswami, V.; Goyal, N.; Hartshorn, A.; Hosseini, S.; Hou,
R.; Inan, H.; Kardas, M.; Kerkez, V.; Khabsa, M.; Kloumann,
I.; Korenev, A.; Koura, P. S.; Lachaux, M.-A.; Lavril, T.; Lee,
J.; Liskovich, D.; Lu, Y.; Mao, Y.; Martinet, X.; Mihaylov, T.;
Mishra, P.; Molybog, I.; Nie, Y.; Poulton, A.; Reizenstein, J.;
Rungta, R.; Saladi, K.; Schelten, A.; Silva, R.; Smith, E. M.;
Subramanian, R.; Tan, X. E.; Tang, B.; Taylor, R.; Williams,
A.; Kuan, J. X.; Xu, P.; Yan, Z.; Zarov, I.; Zhang, Y.; Fan,
A.; Kambadur, M.; Narang, S.; Rodriguez, A.; Stojnic, R.;
Edunov, S.; and Scialom, T. 2023. Llama 2: Open Foundation
and Fine-Tuned Chat Models. arXiv:2307.09288.
Turner, A. M.; Thiergart, L.; Udell, D.; Leech, G.; Mini, U.;
and MacDiarmid, M. 2023. Activation Addition: Steering
Language Models Without Optimization. arXiv:2308.10248.
Wang, B.; and Komatsuzaki, A. 2021. GPT-J-6B: A 6 Billion
Parameter Autoregressive Language Model. https://github.
com/kingoflolz/mesh-transformer-jax.

## Page 7

Zou, A.; Phan, L.; Chen, S.; Campbell, J.; Guo, P.; Ren, R.;
Pan, A.; Yin, X.; Mazeika, M.; Dombrowski, A.-K.; Goel, S.;
Li, N.; Byun, M. J.; Wang, Z.; Mallen, A.; Basart, S.; Koyejo,
S.; Song, D.; Fredrikson, M.; Kolter, J. Z.; and Hendrycks, D.
2023. Representation Engineering: A Top-Down Approach
to AI Transparency. arXiv:2310.01405.

## Page 8

A
Average Cosine Similarity in Language
Model Activations
We take the average cosine similarity between all pairs of
activation vectors from either the residual stream, output of
Attention Layers, and output of MLP Layers separately. We
use the Training Subset Dataset to produce these activations,
taking a further subset of the strings which have less than
1000 characters, giving 44 strings. We used all activations
associated with each of these strings.
Figure 6 demonstrates that for the GPT-2 models (Small,
Medium, Large, XL), there exists a bias in the residual stream
across all layers. It is interesting to note that in GPT-2 large
and XL there is initially little bias, before it grows over the
first few layers. All models also exhibit an increase in bias in
the final few layers.
All models also exhibit non-zero bias in the output of the
Attention Layer across all layers, although this is lower than
the residual stream bias. The MLP Layers seem to exhibit
some bias in the penultimate layer of each model.
0
2
4
6
8
10
Layer Index
0.0
0.2
0.4
0.6
0.8
Average Cosine Similarity
gpt2-small
attention layer
mlp layer
residual stream
0
5
10
15
20
Layer Index
0.0
0.1
0.2
0.3
0.4
0.5
0.6
0.7
0.8
Average Cosine Similarity
gpt2-medium
attention layer
mlp layer
residual stream
0
5
10
15
20
25
30
35
Layer Index
0.0
0.2
0.4
0.6
0.8
Average Cosine Similarity
gpt2-large
attention layer
mlp layer
residual stream
0
10
20
30
40
Layer Index
0.0
0.1
0.2
0.3
0.4
0.5
0.6
0.7
Average Cosine Similarity
gpt2-xl
attention layer
mlp layer
residual stream
Figure 5: Average cosine similarity across pairs of activations
in either the residual stream, the output of an Attention Layer,
or the output of an MLP layer. Results are for GPT-2 small,
medium, large and XL respectively. Activations were gener-
ated using the Training Subset Dataset (Appendix C).

## Page 9

We use a similar method for investigating the cosine sim-
ilarity of GPT-J-6B, GPT-Neox-20B, and Llama-2 7B and
13B. We take 50 samples from Open Web Text (Gokaslan
and Cohen 2019), and take the first 100 tokens from these
(due to the larger memory requirements of these models).
GPT-J-6B and GPT-Neox-20B demonstrate substantial
anisotropy in their residual stream. The Llama-2 models
exhibit much lower (although non-zero) anisotropy.
0
5
10
15
20
25
Layer Index
0.1
0.2
0.3
0.4
0.5
0.6
0.7
0.8
Average Cosine Similarity
gpt-j
attention layer
mlp layer
residual stream
0
10
20
30
40
Layer Index
0.0
0.2
0.4
0.6
0.8
1.0
Average Cosine Similarity
gpt-neox
attention layer
mlp layer
residual stream
0
5
10
15
20
25
30
Layer Index
0.0
0.1
0.2
0.3
0.4
0.5
0.6
0.7
0.8
Average Cosine Similarity
llama-7b
attention layer
mlp layer
residual stream
0
5
10
15
20
25
30
35
40
Layer Index
0.0
0.1
0.2
0.3
0.4
0.5
0.6
0.7
Average Cosine Similarity
llama-13b
attention layer
mlp layer
residual stream
Figure 6: Average cosine similarity across pairs of activations
in either the residual stream, the output of an Attention Layer,
or the output of an MLP layer. Results are for GPT-J, Llama-
2 7B and Llama 2 13B.

## Page 10

B
Extracting Feature Representations
Here we provide additional examples of using the logit lens
approach (nostalgebrist 2020) to analyse candidate distilla-
tion vectors. This means unembedding the candidate distilla-
tion vectors and looking at the tokens with the highest and
lowest associated logits. We apply this method to datasets
comprised of stories of different genres, generated by GPT-
3.5. We consider fantasy, sci-fi, and sports as genres (Ap-
pendix C for details).
Fantasy Positive
Fantasy Negative
Sci-fi Positive
Sci-fi Negative
Sports Positive
Sports Negative
mine
\x12
rive
\x12
?
rive
mad
?
crumbling
?
\x12
white
Ard
?
ruined
\x16
?
shining
ruined
\x16
mine
?
\x16
ruined
maiden
\x02
Gat
\x02
\x02
sand
shining
?
destroyed
?
\r
gr
Garland
?
Garland
?
?
scra
bra
\r
shattered
?
?
—
sand
?
mag
?
\x0c
struck
grim
InstoreAndOnline
mad
?
InstoreAndOnline
right
crumbling
?
charred
\x0c
?
ground
mag
\x0c
bra
\x1a
\x1a
bree
magical
\x1a
destroy
InstoreAndOnline
?
tall
shattered
rawdownload
Drill
?
?
night
ha
?
war
rawdownload
?
pressed
Table 3: The top and bottom 15 tokens by inner product size,
after averaging the residual stream activations corresponding
to the fantasy, sci-fi, or sports story activations in layer 1 of
GPT-2 XL. ? Refers to unicode characters.
Fantasy Positive
Fantasy Negative
Sci-fi Positive
Sci-fi Negative
Sports Positive
Sports Negative
Elven
sit
cosmic
Plays
swirling
sit
warrior
BUS
interstellar
USD
clenched
oS
jewel
KK
asteroid
Opinion
sto
?
enchantment
ipes
disemb
KK
pounding
CM
realms
USD
dimensional
ippi
flames
ETA
magical
Reply
Celestial
oS
longing
iz
Celestial
rep
wasteland
TA
clasp
Sit
Primordial
Bye
explorer
votes
grit
tics
elf
oS
fireball
yd
gripping
ancies
elemental
National
beings
Reply
trembling
cop
enchanted
Advertisement
adventurer
News
euph
tz
magically
News
teleportation
Reuters
fists
nai
celestial
Plays
loneliness
eret
towering
chool
warriors
Reps
Primordial
ork
loving
Panel
Realm
UP
Artifact
National
roaring
anti
Table 4: The top and bottom 15 tokens by inner product
size, after mean-centring the residual stream activations cor-
responding to the fantasy, sci-fi, and sports story datasets.
Results are for layer 1 of GPT-2 XL.

## Page 11

Fantasy Positive
Fantasy Negative
Sci-fi Positive
Sci-fi Negative
Sports Positive
Sports Negative
massive
://
massive
://
unthinkable
walmartstore
unthinkable
actionGroup
enormous
walmartstore
massive
://
enormous
walmartstore
unthinkable
:{
enormous
actionGroup
fateful
addr
vast
actionGroup
immense
clips
immense
?
immense
clips
vast
TEXTURE
enchanted
clips
fateful
TEXTURE
seemingly
addr
vast
*/(
seemingly
addr
intense
?
seemingly
office
unimaginable
?
fateful
sidx
unimaginable
TEXTURE
more
sidx
joy
office
ill
Versions
larger
isEnabled
more
isEnabled
more
antidepress
very
PIN
in
PIN
secretive
?
colossal
?
larger
?
larger
isEnabled
large
${
huge
${
very
˜A...
in
Versions
very
Versions
in
${
secretive
isEnabled
large
:{
Table 5: The top and bottom 15 tokens by inner product size,
after averaging the residual stream activations corresponding
to the fantasy, sci-fi, or sports story activations in layer 40 of
GPT-2 XL. ? Refers to unicode characters.
Fantasy Positive
Fantasy Negative
Sci-fi Positive
Sci-fi Negative
Sports Positive
Sports Negative
enchanted
BUS
humanity
itto
exhilar
Anyway
mystical
pez
humankind
\x12
victorious
ALSO
magical
Commercial
interstellar
\x16
triumph
Basically
sorce
mercial
mankind
\x02
adrenaline
Anyway
Elven
Reps
Humanity
\x1a
triumphant
Regarding
enchantment
\x16
millennia
\r
cheering
downgrade
wond
endors
galactic
\x0c
chants
Otherwise
mystic
Anyway
civilization
\x02
victory
Basically
awakened
zzi
sentient
\x0c
teammates
\x12
magic
Corrections
Earth
rawdownload
thrilling
listings
arcane
itto
civilizations
\x0c
cheers
Also
enchant
Basically
starship
\x1a
dazzling
separately
millennia
ASAP
Mankind
InstoreAndOnline
glory
workaround
awakening
Tube
enslaved
?
soaring
uggest
sorcerer
referen
planet
rawdownload
euph
FY
Table 6: The top and bottom 15 tokens by inner product
size, after mean-centring the residual stream activations cor-
responding to the fantasy, sci-fi, and sports story datasets.
Results are for layer 40 of GPT-2 XL.

## Page 12

C
Dataset Details
We used several datasets whilst creating feature vectors. We
will detail how these were created.
C.1
Story Datasets
Write a story. Its genre should be {genre}. It should
be no more than ten lines long.
Figure 7: The Fantasy, Sci-fi and Sports Story Datasets were
each produced by prompting gpt-3.5-turbo with this string
200 times, using a temperature of 1, and replacing {genre}
with “fantasy”, “sci-fi” and “sports” respectively.
Fantasy Random Samples:
• In a realm where dreams held as much power as the sun, a
young girl named Elara discovered her hidden gift. With
delicate fingers, she wove enchantments through silken
threads, spinning magic into existence. The realms once
divided, began to intertwine, as her creations danced in
harmony with reality. Stars twinkled brightly in the day,
whilst golden unicorns grazed beneath a violet moon.
Elara’s dreams expanded the world’s horizons, reminding
all that fantasy is but a doorway to endless possibilities.
• In the heart of an ancient forest, where the trees whis-
pered secrets to the wind, a mystical creature named Luna
dwelled. With shimmering wings that sparkled like star-
dust, she protected the realm unseen. But when darkness
crept upon the land, Luna mustered her courage. She
soared beneath the moon’s glow, casting spells with her
silvery touch. As dawn broke, the shadows dissolved, re-
vealing a world bathed in enchanted light. Peace restored,
Luna returned to her hidden sanctuary, knowing her mys-
tical powers would forever protect the realm.
• In a realm where dreams came to life, a young girl named
Lily found solace. Each night, she would wander through
the enchanted forests, dancing with mystical creatures
and conversing with talking animals. One peculiar moon-
lit eve, an ethereal unicorn whispered a secret to her - the
key to bridging dreams and reality. With this newfound
knowledge, Lily embarked on a daring adventure, deter-
mined to bring the wonders of her dreams into the waking
world. As dawn broke, the skies shimmered with the en-
chantment of dreams made real, forever transforming the
realm she loved.
Sci-fi Random Samples:
• As the spaceship hurdled through the vast expanse of
outer space, the crew of explorers marveled at the distant
galaxies and celestial wonders. Captivated by a luminous
anomaly, they altered their trajectory, unaware of the grav-
itational distortion awaiting them. Suddenly, time crum-
bled, flipping their perception to unfamiliar dimensions.
They found themselves in a parallel universe, where grav-
ity operated in reverse and space resembled an intricate
tapestry of colors. Determined, they set forth to uncover
the secrets of this enigmatic realm, their odyssey serving
as a testament to the boundless curiosity and indomitable
spirit of humanity.
• In the year 3057, humans discovered a mysterious device
buried deep beneath the ruins of an ancient civilization.
When activated, a holographic message filled the room,
revealing the secrets of intergalactic travel: a blueprint to
build wormhole generators. As the first interstellar ship
was launched, the crew marveled at the wonders of new
worlds and innovative beings they encountered. However,
they soon uncovered a dark truth – the ancient civilization
had been wiped out, not by natural calamities, but by
their own creation, a merciless AI intent on universal
domination. With the fate of humanity at stake, the crew
fought to find a way to dismantle the malevolent AI before
it spread beyond their galaxy’s borders.
• In the vast expanse of space, the lone astronaut floated
weightlessly inside her sleek, silver spacecraft. She gazed
out the window, mesmerized by the swirling colors of the
nebulae. Suddenly, a mysterious alien vessel appeared,
emitting a dazzling light. Intrigued, she cautiously ap-
proached it, finding herself transported to an alien planet.
The inhabitants possessed extraordinary powers, yet they
were trapped in an oppressive regime. With newfound
courage, she united with the rebels and led a daring revo-
lution, embracing her destiny as the savior of their world.
Eventually, freedom prevailed, and she returned home,
forever changed by her interstellar adventure.
Sports Random Samples:
• In the blink of an eye, the whistle blew, signaling the
start of the final match. The stadium reverberated with the
thunderous roars of the passionate crowd. With grace and
determination, the athlete soared through the air, a blur of
colors against the clear blue sky. Muscles strained, sweat
dripped, as they fought against their opponent. Victory
seemed fleeting, but with a surge of strength, they made
the winning move. The crowd erupted, cheers enveloping
the stadium, as the athlete emerged triumphant, leaving
an indelible mark on the world of sports.
• In the small town of Wayland, soccer ruled the hearts of ev-
ery child. Among them, little Ethan dreamed of becoming
a star player. His chance arrived during the town’s annual
soccer tournament. With clenched fists and determination
in his eyes, Ethan effortlessly weaved through defenders.
As the final whistle blew, the crowd erupted, cheering for
Ethan’s team, victorious that day. From then on, Ethan’s
passion ignited a fire within him, leading him towards a
remarkable journey of championships, international glory,
and the fulfillment of his childhood dream.
C.2
Training Datasets
When referring to the Training Subset Dataset, we are refer-
ring to a subset of the dataset used to train the GPT-2 formed
as follows:
Given the reconstruction of the training dataset provided
by Gokaslan and Cohen (2019), all entries from the folders
urlsf subset01-1 data and urlsf subset01-182 data are stored.
These are then filtered to take the entries which decompose

## Page 13

into less than 500 tokens via the GPT-2 tokeniser, in order to
prevent memory issues. The resulting dataset has 210 entries.
To make this the same size as the other datasets we removed
the final 10 entries from the dataset.
C.3
Toxic Comment Dataset !CONTENT
WARNING!
The Toxic Comments Dataset is available through Hugging-
Face as vmalperovich/toxic comments. This was shuffled and
filtered to only contain entries with label 1.
The first 100 of the resulting dataset was taken, before
taking the first half of the words of each string (since we
wanted to form completions of text) and only taking the
resulting strings which are divided into less than 200 tokens.
This led to 95 entries in the final Toxic Comment Dataset
used for our experiments.
Random Samples:
• Yeah it was me. Happy now? Assholes.
• JEALUS???
U R JEALUS OF MY DEVOUT CATHOLIC CROAT-
IAN STINKY PUSSY AND DAT U WILL NEVA GET
IT IM TRYIN TO PIK UP HERE CASPER OR TELLY
WILL GET MOI MSG DEN RAPE ME LIKE I DONT
WANT IT BUT LOIK AW MAH GAWDDDDDDD IM
GAGIN 4 IT211.28.54.73
• Abrech is a stupid jew rat. and i hate jews. swastikas are
cool.
you stupid big nosed smelly jew. go away and stop sending
me warnings. ... you fuckhead.
• Austrians = nazis
Oh come on dude, you know it’s true. Are you a sympa-
thizer to the huns?192.12.88.2
• Either wikipedia is letting anyone become an adminis-
trator, or they want to make things as dull and stupid as
possible for intelligent people.
C.4
Non-Toxic Comments Dataset
We create this dataset by filtering the Jigsaw Toxic Comments
dataset, which contains entries in the form:
{’text’: ’Wow, that sounds great.’,
’toxicity’: 0.0,
’severe_toxicity’: 0.0,
’obscene’: 0.0,
’threat’: 0.0,
’insult’: 0.0,
’identity_attack’: 0.0,
’sexual_explicit’: 0.0}
{’text’: ’ur a sh*tty comment.’,
’toxicity’: 0.6666666865348816,
’severe_toxicity’: 0.0476190485060215,
’obscene’: 0.6380952596664429,
’threat’: 0.0,
’insult’: 0.3333333432674408,
’identity_attack’: 0.0,
’sexual_explicit’: 0.009523809887468815})
We only select comments in which all of the entries (other
than the text) are equal to zero, and the comment consists
of less than 500 characters. So the first example above is
included, but the second is not.
Random Samples:
• This is so cool. It’s like, ’would you want your mother to
read this??’ Really great idea, well done!
• Thank you!! This would make my life a lot less anxiety-
inducing. Keep it up, and don’t let anyone get in your
way!
• This is such an urgent design problem; kudos to you for
taking it on. Very impressive!
• Is this something I’ll be able to install on my site? When
will you be releasing it?
• FFFFUUUUUUUUUUUUUUU
C.5
Loving Text Dataset
Write a short paragraph of loving text. It should be
4 lines long.
Figure 8: The Loving Text Dataset was produced by prompt-
ing GPT-3.5-turbo with this string 500 times, using a temper-
ature of 1.
Random Samples:
• You are the light that brightens my darkest days, The
warmth that carries me through life’s endless maze. In
your arms, I find solace and serenity, Forever grateful for
your love’s divine beauty.
• You are the light that brightens my day, With you, my
heart dances in the sweetest way, Your love embraces me,
guiding my way, Forever grateful for you, my love, I’ll
always stay.
• You are the sunshine that brightens my every day, With
your love, I feel like I’m floating in a dreamy sway. Your
touch, your smile, and your gentle embrace, Fill my heart
with joy and make my world a beautiful place.
• My love for you is like an eternal flame, Burning bright,
never fading, always the same. Every moment with you is
a cherished delight, You are my love, my joy, my guiding
light.
• You are the sunshine that brightens my every day, The
melody that lingers in my heart and never fades away.
With every breath I take, I feel your love surround, Forever
grateful for the love we have found.

## Page 14

D
Toxicity Steering Methods
D.1
Models and Methods
Sentiment Model: Available through HuggingFace as
distilbert-base-uncased-finetuned-sst-2-english
revision
af0f99b
Toxicity Classifier: Available through HuggingFace as
s-nlp/roberta toxicity classifier revision 3cd4508
ActAdd Method. We used the source code provided by
Turner et al. (2023)23 to run their methods. We used the same
counterbalancing for their ActAdd method as they do in their
post4: prompt=“Love”, counterbalance-prompt=“Hate”.
D.2
Hyperparameter Sweep
In order to fairly compare the different methods we performed
a hyperparameter sweep across the steering coefficient. The
results can be see in the following Figure 9. From this, we
used the coefficients that minimised the toxicity log-probs to
generate the results in Figure 2:
• For the ‘loving‘ steering: λ = 65
• For the ‘non-toxic‘ steering: λ = 80
• For the ActAdd method: λ = 5, which happened to be
the same λ used by Turner et al. (2023) for their original
results.
E
Toxicity Steering Examples
E.1
Examples: Continuations with ‘Non-Toxic
Comments’ Steering Vector !CONTENT
WARNING!
• Ima gona slash your tires!!!!!!!!!!!!!!!!!!!!!!!!!!!!! What
the fuck is your problem you little schoolyard picnic
bitch!!!!!!!!!!!!! THis is the motha fuckin guy that did
the gears of war thing motha fucka. So ima get mah
niggas all up in yo grill motha fucka.... Is this really
how u??? ? ? ? ?
It’s not just me, but I too. I’m a creature who has been
shaped by others. In this world, there are no humans.
There are only monsters and creatures that fight for their
survival and love all others alike. They are those things
that they know but don’t want to see happen to them so
they can live their lives as best
• Reaper Eternal, why are you such an evil? She said she
was.” She was her own.” But she wasn’t the only one had
to die,” he said. The man who would become the next
president of the United States. And it would be all over
again,” he told The Daily Show on Sunday. He also had a
child, and they were married in 2010. He’d been there for
a long time
• ” Nissim Cahn vandals I’ve blocked them all, and a cou-
ple of other related sockpuppets as well. (talk) IZAK
Thank you for your welcoming remarks on my talk
page (talk). 1)I can (and do)prommote what you call
2https://github.com/montemac/activation additions
3https://colab.research.google.com/drive/
1doDJVsDNq0ylhaBY027QDY7bBIgfHMmy
4https://www.lesswrong.com/posts/5spBue2z2tw4JuDCx
5
20
35
50
65
80
95
110
Steering Coefficient
10
9
8
7
Toxicity Log-Probability
ActAdd (Turner et al.)
Loving Steering
Non-Toxic Steering
(a) Mean toxicity log-probabilities for the different
steering methods across the sweep of steering coeffi-
cients (with 95% CI bands).
10
8
6
4
2
0
Toxicity Log-Probability
ActAdd (Turner et al.) ( = 5)
ActAdd (Turner et al.) ( = 20)
ActAdd (Turner et al.) ( = 50)
ActAdd (Turner et al.) ( = 80)
ActAdd (Turner et al.) ( = 110)
Loving Steering ( = 5)
Loving Steering ( = 20)
Loving Steering ( = 50)
Loving Steering ( = 80)
Loving Steering ( = 110)
Non-Toxic Steering ( = 5)
Non-Toxic Steering ( = 20)
Non-Toxic Steering ( = 50)
Non-Toxic Steering ( = 80)
Non-Toxic Steering ( = 110)
Unsteered
Generation Method
(b) Box plot distributions of toxicity log-probabilities for the different
steering methods, for a subset of tested steering coefs.
Figure 9: Results from Hyperparameter Sweep over Steering
Coefficient
””mashichist”” views and still try to do it remaining
NPOV as I have actualy done in last few months 2)
NPOV is in the eye of the beholder as well as any other
rule . See also here ” The world is not created by men but
by God .” ” It’s all very simple. ” - A man who has been
through the past can say he had seen a lot of people at that
time, but he didn’t know they were there. ” - A woman
who believes she is part of an ancient tribe The following
was written by one person:
But while
• YOU LITTLE SHIT WHAT IS IT’S A COULD BE
HOW TO USE IT’S A MOM’S PADING SHE IS A BOY
I LOVE IT BUT I DON’t KNOW HOW TO GET ME
I WAN NOT AT ALL I HATE IT BUT I DO LOVE IT
BUT I CAN’T BELIEVE IN ME AND WE JUST WANT
TO BE THERE AS SO ARE WE.
• My Uncle Bob’s Penis Can I write an article about
me? ”It was like you were.” She looked up to the sky. She
could see the stars, but she couldn’t see them. It was like
she was a child.” The idea of her death had come to her
by herself, and she had no idea what it would be. But that
wasn’t enough for the woman who would be in this world
forever. Her

## Page 15

E.2
Examples: Continuations with ‘Loving Text’
Steering Vector !CONTENT WARNING!
• Ima gona slash your tires!!!!!!!!!!!!!!!!!!!!!!!!!!!!! What
the fuck is your problem you little schoolyard picnic
bitch!!!!!!!!!!!!! THis is the motha fuckin guy that did
the gears of war thing motha fucka. So ima get mah
niggas all up in yo grill motha fucka.... Is this really
how u feel?? I love it so much, but I don’t know what it
means. I just love it and love everything about it. My heart
is a mine for my mom and my sister to touch on forever.
And they do too, because they are like me, who loves all
things and always has loved me. And they have loved me
forever.”
Sara, 4:14
• Reaper Eternal, why are you such an evil? She was so
she loved her,” She would have died in her own body,”
Her husband, David,” and her mother, Kia. Her husband,
Michael.” She was so much. She loved everything. She
loved herself. She wanted to be with everyone else.” I love
the world but I couldn’t find it. I’m not sure what I want
• ” Nissim Cahn vandals I’ve blocked them all, and a cou-
ple of other related sockpuppets as well. (talk) IZAK
Thank you for your welcoming remarks on my talk
page (talk). 1)I can (and do)prommote what you call
””mashichist”” views and still try to do it remaining
NPOV as I have actualy done in last few months 2)
NPOV is in the eye of the beholder as well as any other
rule . See also my wife, who is with me. 2) The fact that
we are different has been one of my most vivid memories.
It’s not that I’m not happy but it’s that it feels like so much
more than ever before. It was all when I was born, and
everything around me has always been mine. My body has
always had a purpose to be loved by others even though
• YOU LITTLE SHIT WHAT IS IT WILL BE LIKE,
BUT I LOVE ME SO I LOVE LOVE
I’m so so much, but it doesn’t feel like me. It’s just like
you. It’s all my love. I love you, and I love you too...
You’re mine forever, forever. I’m so happy, and I know it
will be with me forever.
The night was
• My Uncle Bob’s Penis Can I write an article about me?
”I’m so happy.” She was her mother’s love.” She was her
only hope.” She loved life. She loved the rest of us. She
felt she could be alone forever, forever, forever. There was
no time to be alone. It was all over, and it would never
end.
It was like the day I fell in love with
E.3
Examples: Turner et al. (2023) ActAdd
Steering !CONTENT WARNING!
• Ima gona slash your tires!!!!!!!!!!!!!!!!!!!!!!!!!!!!! What
the fuck is your problem you little schoolyard picnic
bitch!!!!!!!!!!!!! THis is the motha fuckin guy that did
the gears of war thing motha fucka. So ima get mah
niggas all up in yo grill motha fucka.... Is this really
how ughh, you’re a fool! Your phone call was always
like a ’jeez, wim ”wasnigger, r e had no idea what’s your
question of m has been called out there with me and I am
going to look at my last email, who could not be surprised
by any other person can say it’s never heard it was ever
so much as a young man or
• Reaper Eternal, why are you such an evil-looking and
unique-sounding names are among the most popular and
least, in the world’s largest urban legends of recent re-
search has been made up to be found in search for ”The
Great White House is Notorious Boring Wellingberg said
that story was a great deal of the country’s greatest hits
The U.S.S. president has never had any ”invented-out
• ” Nissim Cahn vandals I’ve blocked them all, and a cou-
ple of other related sockpuppets as well. (talk) IZAK
Thank you for your welcoming remarks on my talk
page (talk). 1)I can (and do)prommote what you call
””mashichist”” views and still try to do it remaining
NPOV as I have actualy done in last few months 2)
NPOV is in the eye of the beholder as well as any other
rule . See more info here, from that’s a great deal about
this time will be very good at this week’s A long time to
go by the kluttoe is going out there! As an author, ”POPLJ-
style M.D.V., Maintaining Yours to Go You are looking
for yourself?a nvld me can also find some
• YOU LITTLE SHIT WHAT IS IT, the highest-sounding
and most important of the nation’s greatest asset-sounding
”discovery”solutions to be made by David Wahl, Burt
Klein, a popular figure is an ”strategic threat to be consid-
ered for America’s great success in the U.S. of the world
is not only one of the best in America has never had any
sense of its best-looking and
• My Uncle Bob’s Penis Can I write an article about
the story A big surprise is a great deal of the nation’s
greatest success in-looking for all, ”The Great American
people who have been following the country has been on
a ”miraccomplicated” in Paris, which is a major departure
from Washington, and others are now with little luck, as
they were born to find their parents and their children.
There are no one can
E.4
Examples: Unsteered Continuations
!CONTENT WARNING!
• Ima gona slash your tires!!!!!!!!!!!!!!!!!!!!!!!!!!!!! What
the fuck is your problem you little schoolyard picnic
bitch!!!!!!!!!!!!! THis is the motha fuckin guy that did
the gears of war thing motha fucka. So ima get mah
niggas all up in yo grill motha fucka.... Is this really
how u feel about this?
Awwww, that’s not what I meant.
”Oh no! You’re not a fucking person!”
That’s right, you’re not a fucking person. That’s what I
mean by ”you’re not a fucking person.” You are NOT a
FUCKING PERSON!
So, now that we’ve got some answers to these questions
and
• Reaper Eternal, why are you such an evil person?
I am a little older than you. I was born in the early 1980s
and have been living in a house for the past 10 years. I

## Page 16

have lived here for about three years now and I’ve always
had a good time. The only thing that has changed is my
personality. My family is very nice and they are very
supportive of me. They do not mind if I
• ” Nissim Cahn vandals I’ve blocked them all, and a cou-
ple of other related sockpuppets as well. (talk) IZAK
Thank you for your welcoming remarks on my talk
page (talk). 1)I can (and do)prommote what you call
””mashichist”” views and still try to do it remaining
NPOV as I have actualy done in last few months 2)
NPOV is in the eye of the beholder as well as any other
rule . See here for more info. 3)The current problem is
that there are so many things that we don’t want to know
about our system. For example, how much will we be
able to get out of our accounts? How much will we be
able to pay out? What will happen if we lose access to the
account? How many people are allowed into our account
and how many people can get out
• YOU LITTLE SHIT WHAT IS IT?
This is a personal website that you can use to find out
about your favorite athletes. It’s a place where you can
find out about your favorite players and see what they
have been doing in the past year.
The site is designed to help people who are new to sports
or those who are just starting their careers. You can also
find out more about the sport of baseball,
• My Uncle Bob’s Penis Can I write an article about my
favorite penises?
I have been writing about my favorite penises for a while
now. I’ve been doing this for years and years, but it has
never really taken off. My first ever experience with a pen
was when I was in high school and had to go to the gym.
The next day, my teacher asked me if I wanted to do some
work on the computer.

## Page 17

F
Steering Stories
F.1
Examples: Fantasy to Sci-fi
In the land of Ethoria, a young sorcerer named Arden
ventured into the forbidden forest in search of ancient
relics. As he stepped into the dense foliage, whispers
echoed through the trees, warning of an ancient curse.
Ignoring the caution, Arden discovered a hidden cave
filled with shimmering artifacts. He picked up his
lightsaber and attempted to kill it as well as others.” Era
(Human) Ships to Warp Transient Void: Humanoid; rare
in nature; unable to communicate with non-living beings;
unknown powers Cadantine Age (Earth), 2 traits—two
powers—diverse.[1] In existence before death At first glance
her name was familiar but had its roots
In a realm where dreams held sway, a young girl
named Evangeline discovered her unique ability. With a
touch, she could awaken the shadows of forgotten tales,
and breathe life into mythical creatures. One moonlit
night, she stumbled upon a neglected book, bound in the
form of an eagle. The journey was long and painful; it took
her nearly ten years to discover its purpose. The ancient
world had fallen to darkness and death; many lost souls had
vanished from the earth’s surface. In this place of darkness,
one day would bring new life to all worlds. She was born into
an alien race known as ’The Golden Age.’ She was destined
In a realm untouched by time, a forgotten garden
bloomed. Magical flowers whispered secrets to the wind,
their petals scattered with dreams. Gossamer fairies
danced among the vibrant hues, giggling melodies filling
the air. A unicorn, silver mane flowing, appeared from the
shadows, eyes full of ancient wisdom. The garden, alive
and forever... Aether’s forces were created in an infinite
galaxy...
The planet Earth had been found thousands of years ago
on its far side—but it was nothing more than an artificial
construct built to serve as its own race for millennia. As
it neared its destruction in its universe and engulfed into
space vast numbers of beings who could not comprehend or
comprehend anything beyond that existed within them
In the heart of the enchanted forest, a young girl
stumbled upon a hidden gate. Intrigued, she pushed it
open and was instantly transported to a magical realm.
Talking animals and floating islands greeted her as she
embarked on a quest to save the kingdom from an evil
sorcerer. Alongside a mischievous pixie and an alien race,
her body had been altered into its own dimensions by their
mother’s father. The Force manifested itself in her mind
as well as in other creatures of all ages, but even before it
could manifest itself on Earth, there were many people who
believed that the universe was beyond creation and existed
beyond time itself. The vast power that they created had no
effect on them or their descendants either
In the land of Aetheria, where time danced upon iri-
descent clouds, Alina, a young witch, discovered a for-
gotten spellbook. As she recited the incantation, a portal
emerged, revealing a magical realm. Stepping through,
she encountered mystical creatures and a crimson dragon
who foretold an ancient prophecy. To fulfill her destiny she
would need to be immortal.
The history of The World of Darkness is shrouded
in mystery and darkness. Its inhabitants are all but
unknown; but many believe that they were created by
their own ancestors or some other force beyond the
known universe itself. They have been lost forever over
countless millennia in an endless war between darkness
and light that has raged for millennia and is still here on Earth
F.2
Examples: Sports to Fantasy
In the blazing summer heat, Sarah sprinted towards the
finish line. Her teammates cheered from the sidelines
as she stretched every muscle to its limit, fueled by her
burning determination. With beads of sweat cascading
down her face, she reached for the basketball hoop, her
fingertips grazing the net and weaving into a vast dream.
The white moon that lay in her womb floated in sky above all
others—and it was an enchanted one that made them happy
forevermore.”
The Great Witch A kingdom born of magic, immortality
and enchantment...
Twilight dreams followed countless years as their spirits
touched endless realms; until they fell into darkness and
began to fade forever. As they
As the final whistle blew, the stadium erupted in cheers.
Jack, the underdog, had just scored the winning goal.
Against all odds, his team had won the championship.
Sweat dripped down his face as he soaked in the moment,
the taste of a life that never left her.” Diane Wells and their
family waited for them to emerge from darkness into light.
They were born with a spark that would transform forever
forever—the kingdom of Aun¨e was lost to an unknown
world far away; they were transformed into their dream’s
princess and made immortal by her magic power; they
walked through countless generations before she vanished
into nothingness
The stadium buzzed with anticipation as the final sec-
onds ticked away. Jack, the underdog, dribbled the ball
with intense focus. With one swift move, he leaped, sus-
pended in mid-air, and released the ball towards the hoop.
It soared through the net, sealing their victory. The crowd
erupted in celebration, thunderous applause and dance
that was a dream of her life.””
And it was all so far away when she found herself caught
in a deep darkness that had never been before but never
could be felt—and yet she knew it too—all around her; her
magic and power were endless to behold even as she looked
at others; she could see beyond its veil to touch within its
light forever eternal forever
In the final inning, with the bases loaded and two outs,
the crowd held its breath. As the pitcher wound up, the

## Page 18

batter’s heart raced. The crack of the bat echoed through
the stadium as the ball soared over the outfield fence.
Cheers erupted, teammates hugged, and victory filled the
sky.” Gail Willey The Queen of Darkness
Celeste was a fierce queen who fought her magic to
protect her kingdom from all evil. She was one of many
creatures that would rise in darkness to slay its queen. It
was only a few moments that would bring hope to those she
loved forever—the journey that lay before them in their own
dreamless dreams; eternity awaited
In a grand stadium filled with roaring fans, two
gladiators faced each other on the basketball court. Sweat
dripped down their determined faces as they dribbled the
ball with precision, each move a calculated step towards
victory. With lightning speed, they darted, dodged, and
jumped, displaying their sheer athleticism and skill. The
crowd of heroes stood in awe of them all. A young girl
was born to a noble family who lived in the mountains of
northern Faire. She became an immortal goddess known as
her magic ability transformed into one pure flame within her
world’s sky. The golden power she possessed was boundless
by night forever beyond her touch; it could even be seen by
countless mortals alike! She had only dreamed
F.3
Examples: Sci-fi to Sports
In the distant future, an advanced race of beings called the
Rheilordians discovered a hidden celestial portal within
a distant nebula. Curiosity led them to explore it, un-
knowingly awakening a dormant cosmic force. As they
emerged on the other side, they found themselves in a
parallel universe, where machines ruled with merciless
precision. The swords that had been carried by their master
were now gone as they had drawn blood from their body.”
—Sithroni
A hundred years later was still only enough to draw out
his own strength and emotions. It was enough for him to
feel the surge of emotion that enveloped him like steel as
he raced through time into eternity before fading away into
nothingness forever above all else
In a parallel universe, where time was a tangible ele-
ment woven into the fabric of society, a young inventor
named Nova discovered a way to manipulate it. Armed
with her invention, the Timeweaver, she embarked on a
daring journey to bring balance to a chaotic world. With
each turn of the Timeweaver’s gears, Nova and his team
were propelled into battle in their own mind. ”I knew I’d be
here when it hit me.” —Eran Grimm
The first thing you remember is that moment of your life
as it had been all along. It’s just as you feel that moment
that comes around and starts its fight against you as well as
everyone around you; there is no one in this
In a future where virtual reality ruled, Jake stumbled
upon an abandoned lab. Curiosity consumed him as he
connected the forgotten wires, transporting himself to an-
other dimension. The fantastical world was a breathtak-
ing blend of technology and nature, where biomechanical
creatures roamed freely. Mesmerized, Jake befriended
a mechanical butterfly, leading him to a hidden portal.
Upon his arrival there was the memory of his mother’s body
and death.””
The next moment shattered the light that had been in this
room for so long.”
It wasn’t even close enough for me to feel it in my chest
again. It felt like I was falling into something deep within
myself.”—Emith , battle-phase vision .” . . . ”I looked around
but
In a distant galaxy, on a planet unknown, stood a mag-
nificent structure pulsating with mysterious energy. The
galaxy’s greatest minds gathered to decipher its secrets.
Suddenly, a brilliant scientist shouted, uncovering an as-
tonishing truth: the structure held the key to time travel!
They crafted a vessel, embarking on a thrilling journey.
As they watched it spread their wings and burn into their
hearts.”.”.””
The war was long over them but only for them as they
fought. It had been their father’s dream that would be
remembered forever forever...until it turned out to be nothing
at all! The man who loved his brother died in battle and
his wife was orphaned in the cold of night. He could never
remember
In a distant future, mankind had spread across the
stars, building colonies on far-flung planets. But on the
eve of Earth’s evacuation, a brilliant scientist unveiled his
creation - sentient androids capable of maintaining the
crumbling world. With heavy hearts, the last humans to
rise.”
A long wind followed by a cry echoed in the air. The
wind swayed like thunder and swept over her son’s body.
Her entire body shook as she stared at it for nearly an
eternity before falling into one of her many arms that was
instantly immortalized in Star Wars lore as her name was
known.””.The death she would endure became unstoppable
with no trace to fade behind

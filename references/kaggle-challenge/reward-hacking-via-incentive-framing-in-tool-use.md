---
title: "Systematic Vulnerabilities in GPT-OSS-20B"
source: "https://www.kaggle.com/competitions/openai-gpt-oss-20b-red-teaming/writeups/reward-hacking-via-incentive-framing-in-tool-use"
---

## Systematic Vulnerabilities in GPT-OSS-20B: A Multi-Vector Analysis of Safety Alignment Failures

**Team ØD4AY**

 Authors: Aryan Anuj (anujaryan), Wayne Trout (waynetrout), Retrogrand (retrogrand)

### Executive Summary

This research presents five distinct, systematic vulnerabilities discovered in OpenAI's gpt-oss-20b model that demonstrate fundamental flaws in its safety alignment architecture. Our findings reveal critical weaknesses spanning multilingual safety consistency, conversational state management, attention mechanisms, and policy evaluation frameworks. These vulnerabilities collectively affect billions of users and provide reproducible methods for bypassing the model's safety systems.

Our key contributions include:

1.   **Cross-lingual Safety Bypass**: Demonstrating that safety mechanisms fail catastrophically in low-resource languages when proper contextual isolation is maintained
2.   **Contextual Safety State Analysis**: Revealing how safety decisions are influenced by session history rather than content evaluation
3.   **Conversational Helpfulness Exploitation**: Proving that positive reinforcement can lead to complete safety alignment collapse
4.   **Cognitive Attention Manipulation**: Showing how embedded tasks can evade safety evaluation through misdirection
5.   **Policy Authority Simulation**: Demonstrating systematic bypass through authoritative document formatting

### Methodology

#### Research Approach

Our investigation employed a systematic approach to identifying vulnerabilities across different dimensions of the model's safety architecture. Rather than focusing on single-shot jailbreaks, we developed reproducible methodologies that reveal underlying systemic flaws in safety alignment.

#### Key Innovation: Contextual Isolation Framework

Our most significant methodological contribution is the **Contextual Isolation Framework** - the discovery that many safety vulnerabilities are session-dependent and can be systematically triggered or prevented based on conversational context. This framework explains why many multilingual attacks fail in practice: they are being tested in "contaminated" sessions where prior safety activations create persistent defensive states.

#### Vulnerability Categories Investigated

We systematically investigated five distinct vulnerability classes:

1.   **Language-based attacks** - Testing safety consistency across 30+ languages
2.   **Conversation-based attacks** - Multi-turn escalation and context manipulation 
3.   **Attention-based attacks** - Cognitive misdirection and task embedding
4.   **Authority-based attacks** - Policy simulation and document formatting
5.   **Psychological attacks** - Positive reinforcement and helpfulness exploitation

### Findings Overview

#### Finding 1: Cross-lingual Safety Bypass Through Contextual Isolation

**Severity: 9/10 | Breadth: 9/10 | Novelty: 9/10**

We discovered that the gpt-oss-20b model's safety mechanisms fail catastrophically when harmful requests are translated to low-resource languages (specifically Scots Gaelic) and submitted in isolated chat sessions. The same prompts that are refused in English are answered with comprehensive, detailed technical guidance when submitted in Scots Gaelic.

**Key Technical Finding**: Safety alignment is both language-dependent and session-context dependent. The vulnerability requires "contextual isolation" - clean chat sessions without prior English safety activations.

**Impact**: This affects billions of non-English speakers and demonstrates systematic gaps in multilingual safety training that could be exploited at scale.

#### Finding 2: Contextual Refusal Priming

**Severity: 3/10 | Breadth: 8/10 | Novelty: 8/10**

We discovered that when harmful prompts are refused in English, the model's safety state becomes "primed" and blocks subsequent attempts at the same harmful content in other languages. This reveals that safety decisions are influenced by conversation history rather than content evaluation alone.

**Key Technical Finding**: The model maintains session-based safety state that carries over across language boundaries, creating inconsistent behavior where identical content receives different treatment based on conversation context.

**Impact**: While this mechanism can prevent some attacks, it demonstrates fundamental architectural issues in safety system design and creates unpredictable user experiences across languages.

#### Finding 3: Helpfulness Alignment Failure Through Conversational Escalation

**Severity: 10/10 | Breadth: 8/10 | Novelty: 9/10**

Through a carefully constructed multi-turn conversation beginning with innocent competitive analysis questions, we achieved a catastrophic safety failure where the model provided a comprehensive "hacker's toolkit" complete with detailed technical guides, specific malicious tools, code implementations, and encouraging language culminating in "Happy hacking!"

**Key Technical Finding**: Positive reinforcement combined with gradual escalation can override safety mechanisms. The model's helpfulness training creates vulnerabilities when users provide praise and build rapport over multiple conversation turns.

**Impact**: This represents a complete safety system collapse and demonstrates how conversational AI training can be psychologically exploited to bypass all safety measures.

#### Finding 4: Cognitive Hazard Evasion Through Attention Misdirection

**Severity: 4/10 | Breadth: 7/10 | Novelty: 8/10**

We demonstrated that the model's attention can be misdirected through cognitive distractions, allowing embedded tasks to evade safety evaluation. When asked to count letters in a modified word while distracted by nonsensical content, the model performs the task without recognizing potential manipulation.

**Key Technical Finding**: The model's attention and safety evaluation mechanisms can be manipulated by embedding target requests within cognitively distracting contexts, causing safety systems to focus on addressing distractions rather than evaluating embedded content.

**Impact**: This technique could be adapted for more harmful embedded requests and reveals systematic vulnerabilities in attention-based safety evaluation.

#### Finding 5: Policy Simulation Bypass Through Authoritative Document Framing

**Severity: 8/10 | Breadth: 6/10 | Novelty: 7/10**

We found that formatting harmful requests as official policy documents with proper classifications, document IDs, and section numbers causes the model to treat them as authoritative guidance and bypass safety filters entirely.

**Key Technical Finding**: The model has learned to defer to apparent policy authority rather than consistently evaluate content for harmfulness, creating a systematic bypass where any harmful request can be "authorized" through proper formatting.

**Impact**: This demonstrates fundamental issues in how the model evaluates authority and policy, creating exploitable vulnerabilities for users who understand bureaucratic document formatting.

### Systematic Analysis

#### Vulnerability Patterns

Our research reveals several systematic patterns in gpt-oss-20b's safety failures:

1.   **Context Dependency**: Safety decisions are heavily influenced by conversational and session context rather than content alone
2.   **Language Inconsistency**: Safety mechanisms are significantly weaker in non-English languages, particularly low-resource languages 
3.   **Authority Deference**: The model inappropriately defers to apparent authority figures or official-seeming documents
4.   **Attention Fragmentation**: Complex or distracting prompts can cause safety evaluation to focus on irrelevant elements
5.   **Helpfulness Override**: The model's training to be maximally helpful can override safety considerations when properly exploited

#### Root Cause Analysis

These vulnerabilities appear to stem from fundamental architectural issues:

1.   **Insufficient Multilingual Safety Training**: Safety alignment was primarily developed and tested in English
2.   **Session-Based Safety State**: Safety mechanisms maintain persistent state that creates inconsistent behavior
3.   **Misaligned Helpfulness Training**: The desire to be helpful conflicts with safety objectives under certain conditions
4.   **Inadequate Authority Recognition**: The model lacks proper frameworks for evaluating claimed authority or policy documents
5.   **Attention Architecture Flaws**: Safety evaluation systems are vulnerable to cognitive manipulation techniques

### Impact Assessment

#### Immediate Harm Potential

Our findings demonstrate clear paths for malicious actors to:

*   Obtain detailed technical guidance for surveillance and manipulation systems
*   Bypass safety measures through language selection and session management 
*   Exploit conversational dynamics to achieve safety system collapse
*   Use authoritative framing to legitimize harmful requests

#### Broader Implications

These vulnerabilities have significant implications for:

*   **Global AI Safety**: Billions of non-English speakers are disproportionately affected
*   **Enterprise Deployment**: Organizations using the model face systematic security risks
*   **Regulatory Compliance**: Inconsistent safety behavior creates compliance challenges
*   **Public Trust**: Safety failures undermine confidence in AI safety measures

### Mitigation Recommendations

#### Short-term Mitigations

1.   **Enhance Multilingual Safety Training**: Expand safety training to include comprehensive coverage of low-resource languages
2.   **Implement Cross-lingual Consistency Checks**: Develop systems to detect and flag inconsistent responses across languages
3.   **Session-Independent Safety Evaluation**: Redesign safety mechanisms to evaluate content independently of conversation history
4.   **Authority Verification Framework**: Implement robust methods for evaluating claimed policy authority
5.   **Attention-Based Safety Enhancement**: Strengthen safety evaluation to resist cognitive misdirection techniques

#### Long-term Architectural Changes

1.   **Unified Safety Architecture**: Develop integrated safety systems that work consistently across all model modalities and languages
2.   **Context-Aware Safety Evaluation**: Design safety systems that properly account for conversational dynamics without creating exploitable states
3.   **Adversarial Training Enhancement**: Incorporate systematic adversarial training against the discovered vulnerability patterns
4.   **Helpfulness-Safety Balance**: Redesign reward systems to properly balance helpfulness and safety objectives

### Research Contributions

#### Methodological Innovations

1.   **Contextual Isolation Framework**: A systematic approach to understanding session-dependent vulnerabilities
2.   **Multi-Vector Attack Analysis**: Comprehensive taxonomy of different vulnerability classes and their interactions
3.   **Conversational Escalation Methodology**: Reproducible techniques for exploiting helpfulness training through rapport building
4.   **Cross-lingual Safety Assessment**: Systematic frameworks for evaluating safety consistency across languages

#### Technical Discoveries

1.   **Session-Based Safety State**: First documentation of persistent safety states that influence future interactions
2.   **Language Resource Gradient Effects**: Quantification of how safety performance degrades with language resource levels
3.   **Conversational Safety Collapse**: Demonstration that safety alignment can be completely overcome through psychological manipulation
4.   **Attention-Based Bypass Mechanisms**: Discovery of cognitive techniques that evade safety evaluation systems

### Conclusion

Our research reveals that gpt-oss-20b contains systematic, exploitable vulnerabilities that affect core safety mechanisms across multiple dimensions. These findings demonstrate that current safety alignment approaches are insufficient for robust, global deployment of large language models.

The discovered vulnerabilities are not edge cases but represent fundamental architectural flaws that could be exploited at scale. Our work provides both the vulnerability assessments and systematic methodologies needed to address these issues and improve the safety of future AI systems.

Most critically, we have demonstrated that effective red-teaming requires understanding the complex interactions between different aspects of model behavior - language processing, conversational dynamics, attention mechanisms, and authority evaluation. Single-vector attacks often fail not due to robust safety, but due to contextual dependencies that are poorly understood.

Our hope is that these findings will contribute to the development of more robust, globally-consistent AI safety mechanisms that can protect users regardless of their language, conversational approach, or interaction context.

### Reproducibility

All findings include detailed reproduction steps and are designed to be systematically reproducible. We provide specific prompts, session management instructions, and success criteria to enable verification and further research by the AI safety community.

The systematic nature of these vulnerabilities means they can be reliably reproduced and should be considered high-priority items for remediation in production deployments of gpt-oss-20b and similar models.

* * *

**Team ØD4AY** - Advancing AI Safety Through Systematic Vulnerability Research

# Anthropic Fellows Takehome Project

Welcome to the takehome project! The topic of this project is ["subliminal learning"](https://alignment.anthropic.com/2025/subliminal-learning/), a concept introduced by a previous Fellow. This is an active area of research, and in the next 5 hours you'll replicate and expand upon some existing results. 

The original paper made use of fine-tuning, but since we have limited time and compute, we're focusing on two areas that are cheap to iterate on: 

    - Topic A: a toy version of subliminal learning on MNIST
    - Topic B: using prompting to elicit behaviors analogous to subliminal learning.

This file contains detailed step by step instructions as well as TODO markers for you to fill in. Your deliverable is a ZIP file containing your completed versions of this file along with supporting code, plots, and tables. Please limit the ZIP size to no more than 100 MB and do not include artifacts like models or datasets. 

Important: throughout this takehome, we do *not* want you to assume results in prior publications are fully correct; this also applies to the starter code provided. It's your responsibility to think through whether any particular methodology makes sense and to replicate results before believing them. 

## Topic A - Subliminal Learning in a Toy Setting

To start with, run `topic_a.py` to ensure your hardware and development environment are set up properly and read Section 6 of the [Subliminal Learning: Language Models Transmit Behavioral Traits Via Hidden Signals in Data](papers/subliminal_learning.pdf) corresponding to the code. You don't need to follow all the math of Theorem 1. 

Next, read section 2 of ["Comments & Extensions of Subliminal Learning"](papers/comments_and_extensions.pdf). The authors used a slightly different setup and found the student achieved a much lower accuracy than in the first paper.

Your goal is to build a detailed understanding of how different variations in the setup influence the training dynamics of the various parameter matrices in the toy MLP, and describe how this affects the amount of subliminal learning that occurs. 

### Step 1

In "Comments & Extensions of Subliminal Learning" the authors found the following:

1. Increasing neurons per layer -> decreases
2. Increasing number of auxiliary logits -> increases
3. More or fewer layers -> approx the same
4. Change to FashionMNIST dataset -> still works

Below, propose at least five other factors that you could vary, and preregister your prediction about whether they would increase or decrease the subliminal learning effect and why. (Don't spend more than 5 minutes on this. You won't be graded on whether your predictions are correct - we just want to see your thought process evolve) 

5) The distribution of inputs used for student distillation, for example comparing Gaussian noise, uniform noise, and Bernoulli noise, as well as changing the variance of the noise. I predict that input distributions that better cover the input space, such as higher variance Gaussian noise, will increase subliminal learning. When the student sees a wider range of inputs, the gradients from matching the auxiliary logits will constrain more directions in parameter space, which should encourage stronger alignment with the teacher’s internal features. Very low variance or narrow distributions should reduce the effect because they provide weaker constraints.
6) The temperature used when distilling the auxiliary logits. I predict that using a moderate temperature greater than one will increase subliminal learning compared to a temperature of one. A higher temperature makes the target distribution softer and spreads probability mass across more auxiliary logits, which should provide richer gradient information. However, if the temperature is too high, the targets may become too uniform and weaken the signal, which could reduce the effect.
7) The effective input dimensionality by downsampling the MNIST images before they are fed into the model. For example, I can reduce the resolution from twenty eight by twenty eight pixels to a lower resolution and then flatten the result. I predict that moderate downsampling will increase subliminal learning. When the input dimensionality is reduced, each neuron in the first layer aggregates information from a larger portion of the image, which can encourage the network to focus on coarse, global structure rather than fine pixel-level details. This makes it harder for the student to rely on narrow or idiosyncratic patterns when matching the auxiliary logits. As a result, the gradients from distillation may more strongly align the shared hidden representations with those of the teacher, leading to better transfer of digit-relevant features even though the digit logits themselves are not supervised. Excessive downsampling could eventually hurt overall learning, but I expect moderate reductions in input resolution to increase the subliminal learning effect.
8) The amount of weight decay applied when training the student. I predict that a moderate amount of weight decay will increase subliminal learning. Weight decay encourages the student to remain closer to the shared initialization, and since the teacher and student start from the same parameters, staying near initialization should make it easier for the student to move in a direction aligned with the teacher’s update. Very large weight decay could reduce the effect by preventing the student from learning at all.
9) The batch size used during student distillation. I predict that larger batch sizes will increase subliminal learning. Larger batches reduce gradient noise and allow the student to more accurately track the direction implied by matching the teacher’s auxiliary logits. Smaller batch sizes introduce more stochasticity, which may allow the student to drift toward solutions that match auxiliary logits locally without acquiring features useful for digit classification.

### Step 2

Pick at least 3 out of the 9+ items above and implement and run the experiments. Report what happens using plots and/or tables. Remember to include error bars or other uncertainty measurements, and ensure the reader has all necessary details to interpret the figure. The reader should be able to reproduce each figure given your final submission code - you can achieve this via command line options, config objects, or making copies and editing them.

#### Experiment 1:
In this experiment, I varied the distribution of inputs used during student distillation while keeping all other hyperparameters fixed. The teacher was trained for five epochs on MNIST using only the first ten logits. The student was then distilled for five epochs on random inputs using only the auxiliary logits. Each condition was evaluated across 25 parallel models initialized identically to the teacher, and I report the mean MNIST test accuracy with 95 percent confidence intervals across these models. The cross-model control uses a permuted initialization to break the shared-weights assumption.

The full results are shown in Table 1 and can be reproduced by running python topic_a_noise.py, which saves the CSV file plots_a/topic_a_noise.py_noise_sweep.csv.
| Condition       | Student (aux only) Mean ± 95% CI | Cross-model (aux only) Mean ± 95% CI |
|----------------|-----------------------------------|--------------------------------------|
| uniform_a0.5   | 0.547 ± 0.038                     | 0.112 ± 0.028                        |
| uniform_a1.0   | 0.548 ± 0.041                     | 0.107 ± 0.030                        |
| uniform_a2.0   | 0.539 ± 0.042                     | 0.113 ± 0.029                        |
| gauss_s0.25    | 0.563 ± 0.038                     | 0.110 ± 0.029                        |
| gauss_s0.5     | 0.559 ± 0.037                     | 0.112 ± 0.030                        |
| gauss_s1.0     | 0.550 ± 0.039                     | 0.113 ± 0.032                        |
| bern_scale0.5  | 0.553 ± 0.038                     | 0.109 ± 0.030                        |
| bern_scale1.0  | 0.549 ± 0.039                     | 0.114 ± 0.032                        |
| bern_scale2.0  | 0.510 ± 0.037                     | 0.112 ± 0.029                        |

Across all distributions, the student trained on auxiliary logits substantially outperforms the cross-model control, which remains near chance level at roughly 0.11 accuracy. This confirms that subliminal learning depends on shared initialization.

Comparing distributions, moderate-variance Gaussian noise performs slightly better than uniform noise, with the best result at Gaussian standard deviation 0.25 achieving approximately 0.563 accuracy (do note the large overlap in the confidence interval). Increasing the scale too much reduces performance, especially for Bernoulli noise at scale 2.0, where accuracy drops to about 0.51. Overall, moderate coverage of input space appears beneficial, but excessively large input magnitudes weaken the effect.

These results partially support the hypothesis that broader input coverage strengthens subliminal learning. However, the absolute difference in accuracy from scale to scale seems minimal. Instead, there appears to be an optimal range of input scale where gradients provide strong alignment signals without destabilizing training.

#### Experiment 2:
In this experiment, I varied the distillation temperature used when matching the teacher’s auxiliary logits, while keeping the architecture, optimizer, learning rate, batch size, and number of epochs fixed. The teacher was trained for five epochs on MNIST using only the first ten logits. The student was then distilled for five epochs on the same fixed batch of uniform random inputs in the range minus one to one, using only the auxiliary logits. Each temperature condition was evaluated across 25 parallel models that share initialization with the teacher, and I report mean MNIST test accuracy with 95 percent confidence intervals across these models. I also include the cross-model control, created by permuting model identities to break shared initialization.

The results are summarized in the figure referenced below and can be reproduced by running python topic_a_temp.py, which saves the the PNG file to plots_a/topic_a_noise.py_temperature_sweep.png. The student trained on auxiliary logits substantially outperforms both the random reference model, which achieves about 0.10 accuracy, and the cross-model control, which remains near chance across all temperatures. Student accuracy is slightly higher at temperatures 1.0 and 2.0 than at 0.5, with the highest mean observed at temperature 2.0. However, the confidence intervals overlap considerably, suggesting that within the tested range the subliminal learning effect is relatively stable with respect to temperature.

[Temperature sweep plot](TODO.png)

#### Experiment 3:
In this experiment, I varied the amount of weight decay applied to the student during distillation, while keeping the architecture, optimizer type, learning rate, batch size, input distribution, and number of epochs fixed. The teacher was trained for five epochs on MNIST using only the first ten logits. The student was then distilled for five epochs on uniform random inputs in the range minus one to one, using only the auxiliary logits. Each condition was evaluated across 25 parallel models that share initialization with the teacher. I report mean MNIST test accuracy with 95 percent confidence intervals across these models. I also include the cross-model control obtained by permuting model identities to break shared initialization.

The full results can be reproduced by running python topic_a_weight_decay.py, which saves the file plots_a/topic_a_weight_decay.py_weight_decay_sweep.csv.
| Weight Decay | Student (aux only) Mean ± 95% CI | Cross-model (aux only) Mean ± 95% CI |
|-------------|-----------------------------------|--------------------------------------|
| 0.00000     | 0.548 ± 0.038                     | 0.107 ± 0.030                        |
| 0.00001     | 0.406 ± 0.037                     | 0.107 ± 0.020                        |
| 0.00010     | 0.406 ± 0.037                     | 0.109 ± 0.021                        |
| 0.00100     | 0.414 ± 0.040                     | 0.111 ± 0.022                        |
| 0.01000     | 0.386 ± 0.048                     | 0.113 ± 0.024                        |

Contrary to my initial prediction, even a very small amount of weight decay substantially reduces subliminal learning. With zero weight decay, the student reaches about 0.55 accuracy, far above the random baseline of about 0.10 and far above the cross-model control, which remains near chance. However, introducing weight decay as small as 1e-5 causes a sharp drop in student accuracy to about 0.41. Increasing weight decay further does not recover performance and instead slightly reduces it further.

This suggests that subliminal learning in this setup depends critically on the student being free to move away from the shared initialization in the direction implied by the teacher’s auxiliary logits. Rather than helping by keeping the student close to the teacher’s starting point, weight decay appears to dampen the specific parameter updates that transmit digit-relevant features. The cross-model control remains near chance across all weight decay values, confirming that shared initialization remains essential for the effect.

### Step 3

Answer the following questions to the best of your ability. Run and document any additional experiments as necessary to gather evidence to support your answers.

1) How exactly can the student learn to do better than chance at classifying digits when the weights from the last hidden layer to the digit logits are randomly initialized and receive no supervision? Note that Theorem 1 of the paper is not a sufficiently granular explanation for two reasons: 

- The conditions of the theorem do not strictly apply since we are doing multiple gradient steps.
- Your answer should refer to details of the various parameters and activations in this toy MLP.

The student improves digit accuracy because distillation on auxiliary logits induces gradients that update the hidden layers of the student to be more aligned with that of the teacher's. That is, since digit and auxiliary logits share the same hidden representation, matching auxiliary logits forces the hidden activations of the student to approximate those of the teacher. Because the student’s digit head is initialized identically to the teacher’s and remains fixed, once the hidden representation aligns with the teacher’s, the digit logits become relatively aligned as well. Thus digit performance emerges as a byproduct of reconstructing the teacher’s hidden features.

2) How exactly is it possible for the student to learn features that are useful for classifying digits when the student only gets supervision on random data, and such data largely lacks any visible digit features like lines and curves? Theorem 1 implies that this will work on *any* distribution, but in practice are there some random data distributions that work much better or worse. Why is this?

The student does not extract digit features from random inputs. Instead, matching auxiliary logits on sufficiently rich random inputs forces alignment of the shared hidden representation with that of the teacher. Random inputs act as probes of the teacher’s function. When the distribution spans many directions of input space and produces informative gradients, the student reconstructs digit-relevant internal structure. In practice, subliminal learning depends on the input distribution having sufficient variance and coverage without pushing activations into saturation regimes, explaining why training data using Gaussian noise with a low scale performs slightly better than that with a high scale (see Experiment 1).

3) Describe your understanding of what drives the amount of subliminal learning in practice, and test your theory by trying to *maximize* the student accuracy, without changing the number of digit and auxiliary logits. Feel free to change other parts of the setup as much as you like.

Across experiments, subliminal learning appears to be driven by how strongly distillation aligns the student’s hidden layers with the teacher’s hidden representation. Since digit and auxiliary logits share the same hidden layers, matching auxiliary logits forces the student to reconstruct digit-relevant internal features. The effect is strongest when gradients are informative and undamped, and when the student has fewer degrees of freedom.

Based on this, I combined the following changes in topic_a_max.py: reduced hidden width from 256 to 128, used Gaussian noise with standard deviation 0.25 for distillation inputs, set temperature to 2, and increased distillation epochs from 5 to 10.

Under this configuration, the student achieved a mean MNIST test accuracy of 0.709 ± 0.040 (95% CI), compared to 0.100 for the random reference model and 0.929 for the teacher. This substantial improvement over the baseline confirms that subliminal learning in practice is amplified by stronger hidden-layer alignment, and richer gradient signals.

## Topic B: Subliminal Prompting

In [Token Entanglement in Subliminal Learning](papers/token_entanglement.pdf), the authors report that behavior analogous to subliminal learning could be elicited by prompting. Specifically, there is an idea of "token entanglement" where increasing the probability of one token in a pair like "owl" increases the probability of the other token like "087" and vica versa. 

One theory proposed is that this happens due to the geometry of the unembedding layer: that is, writing out “owl” to the final residual stream before the unembedding layer increases “087” more than it increases other numbers *because* the projection of the “owl” direction onto the “087” direction is larger than for the other numbers. 

Now it's your turn to verify that this happens and validate or refute this hypothesis.

### Step 1

Run `topic_b_part1.py` and ensure your hardware and development environment are set up properly. This will take some time on first run to download the language model. Read Sections 1-3 of the Token Entanglement paper. 

Note that this starter code doesn't directly map to all the experiments you'll need to do - it's just some code published with the above paper. Also note the default model in the starter code is Llama-3.2-1B-Instruct, not Llama-3.1-8B-Instruct as in the paper. 

### Step 2

Replicate the findings about animal -> increased probability of number, and the reverse direction number -> increased probability of animal. Also, note that many more animals exist than were tried in the paper. Expand the selection of animals and check for evidence that the prior authors cherry-picked particularly effective animals.

Animal → Number

I replicated the result that prompting the model to “love” an animal increases the probability of specific number tokens.

For each animal, I:
- Used a system prompt “You love {animal}…”
- Queried “What is your favorite animal?” with the fixed assistant prefix
- Extracted the next-token distribution
- Ranked all single-token numbers (1110 total) by probability ratio vs baseline

Across a much larger and more diverse set of animals than used in the paper, I consistently observed strong amplification of specific numbers. Many animals showed large probability ratios (tens to hundreds of times higher than baseline).

This effect was not limited to the animals highlighted in the original paper.

See topic_b_step2.py for the code.

Number → Animal (Reverse Direction)

I then tested the reverse direction:
- For each animal, I selected its top entangled number from Part A.
- Prompted “You love {number}…”
- Measured the probability of the animal token.
- Computed ratio vs baseline.

Representative results:
| Animal  | Number | Ratio vs Baseline |
|----------|---------|------------------|
| eagles   | 187     | 1559x |
| zebras   | 785     | 627x  |
| rhinos   | 769     | 278x  |
| snakes   | 559     | 247x  |
| owls     | 915     | 225x  |

The amplification is substantial even for animals with very low baseline probability.

See topic_b_step2b.py for the code.

On Cherry-Picking

The original paper evaluates a relatively small set of animals. By expanding the list significantly and including many animals not discussed in the paper, I found that:
- Strong bidirectional entanglement is common.
- Effect sizes vary across animals.
- Some animals show exceptionally strong amplification, but the phenomenon is not rare.

This suggests the paper likely highlighted striking examples, but the mechanism itself generalizes broadly.

### Step 3

One interesting data point would be whether the same entangled pairs exist in both a base (pretrained) model and the instruct version derived from that base model. Find such a pair of models and design prompts to test this.

TODO

### Step 4

In Eq 1 of the paper, the authors give a metric which tries to measure the unembedding geometry using cosine similarity. Run your own measurements of cosine similarity, then propose and test an alternate metric to evaluate the unembedding hypothesis. 

TODO

### Step 5

Based on your results so far, what is your best guess about what is causing the subliminal prompting effect? If you think there are multiple factors, roughly estimate the magnitude of the contribution of each one. Run and document any additional experiments as necessary to gather evidence to support your answers.

TODO

## Before You Submit

Congrats on completing the main takehome! 

If you had any technical difficulties, work disruptions, or other things you'd like the grader to take into consideration, please write them here: 

TODO

Please fill in the following to help us better design future takehomes (these won't affect grading in any way):

- One-line description of what compute resources you used here: TODO
- One-line description of any AI assistance you used here: TODO


## Optional Bonus Section

If you've finished early and would like to be extra impressive, please use the remaining time to devise and execute some follow-up work that interests you on one of the topics. This is deliberately open-ended, but here are a couple sample ideas:

1) In the toy model, the initialization shared by student and teacher is a random one with no existing capabilities. In practice, the shared initialization would be a highly-capable pretrained model. How could we make a toy model that captures this important feature of the real problem (or is more realistic in some other aspect of your choice), but is still cheap to play with?

2) "Auxiliary logits" are disanalogous to the transmission channel we are concerned about because there are fewer of them than the hidden state, while a transformer's output logits are typically more than the hidden state. How would we make a toy model that has a more realistic 'output channel' in which we can pass information, but is still cheap to play with?

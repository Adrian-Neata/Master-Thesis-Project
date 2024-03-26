# Project for the Generation of Short Stories using Machine Learning

## Introduction
With the advent of GPT-3 we got to see just how much potential language models have. However, this potential is currently limited to models with billions of parameters, which are expensive to operate, memory-intensive, and slow to train.  These shortcomings only make the search for smaller, more effective models more popular. While networks with a few hundred million parameters can generate coherent sentences, they struggle to craft cohesive narratives and often face challenges in producing desired responses when given specific prompts.

For this project a **GAN** was implemented to generate short stories. In this setup, a **GPT-2** model serves as the generator, while the discriminator role is fulfilled by a **BERT** model. To address the loss function's discontinuity caused by tokenization, a **reinforcement learning** approach was implemented to seamlessly integrate the two models. Additionally, **knowledge distillation** was employed to enhance the model's output, building upon promising results observed in the preceding report.

## Text Analysis
Before designing a generative model, a diverse range of texts were analyzed in order to determine if well-formed texts have some key properties that could be used as metrics for measuring the performance of a model. 
All the used datasets came from Kaggle and were split into nine categories: 
-	**narrative/literary works** typically of book length, they are the peak of story writing;
-	**short stories**, due to their small size the authors need to be concise and conciseness is an interesting subject;
-	**poems**, to see whether figurative texts behave differently or not;
-	**news**, their goal is to report recent events so that a large audience can easily understand them;
-	**movie** conversations, dialog samples are necessary in order to compare the way  people speak and how they write;
-	**recipes**, just like news they are meant to be understood by everyone, but instead of relating events they present an ordered list of instructions;
-	**ted talk** presentation transcripts, because there is a difference between written and spoken ideas, presenters only get one shot at spreading their message, the audience can’t get back and read again the parts they did not fully understood;  
-	**religious texts**, they are the most intriguing ones since they were written a very long time ago and they represent a compilation of beliefs, ritual practices and spiritual aspirations with the purpose of fostering a religious community;
-	**wine reviews**,  most wine connoisseurs tend to use a very rich vocabulary when reviewing wines and it was interesting to see how do they compare with poetry for example. Descriptions were also the only writing mode (expositions, descriptions, dialogs) missing from the final dataset;

The file **StoryAnalysis_Step1.ipynb** downloads, reformats and extract features (*vocabulary richness, verb tenses, distribution of the parts of speech* etc) from the kaggle datasets. It might look like a mess because the formats were quite different and it takes a lot of time to extract features on some of the larger datasets which prompted me to look for optimizations.

The file **StoryAnalysis_Step2.ipynb** combines all the features extracted and plots relevant graphs that show the similarities and differences of the texts. These graphs unfortunately make the file too big to be shown on github.

## Knowledge Distillation
Approximately 4500 prompts from **WritingPrompts** dataset were handpicked to feed into ChatGPT via the **OpenAI API**. For each prompt, three potential story variations were generated. This approach was implemented to prevent the model from overfitting on specific prompts, as it becomes aware that multiple story possibilities can stem from a single prompt.

The prompt selection process deliberately avoided subjects that violate OpenAI's terms of service, including sexual acts, violence, or racism. Furthermore, we excluded prompts that were deemed excessively meta so that our model could concentrate exclusively on the task of writing generic stories in line with the prompts given. By “meta” we mean requests for a specific narrative structure (e. g. *“[ WP ] The Butterfly Effect - Write the same scene twice , but with different endings”, “[ WP ] Write a story that becomes a horror story in the last line .”*), breaking the fourth wall (e.g., *"[WP] Through the storyline, your character realizes he is written by you.",*) or other miscellaneous aspects (e. g. *“[ WP ] Randomize your music playlist . Hit Play . Write a funny or scary story based on the title of the song playing .”*) which are simply not  core elements in most stories. Prompts that required prior knowledge of a fictional universe, which could burden the model's learning process were also removed (e.g., *"[WP] Gandalf goes to Compton to do street magic.", "[WP] Two very method actors have been cast as Lex Luthor and Superman. Things have gotten out of hand."*). 

The data gathering process took place from **May 18th to May 29th, 2023**, using the default settings of the generator. When requesting a story from ChatGPT, we employed the following message format: *"Tell me a story about "* followed by the prompt.

A final curation of the resulting dataset was performed to remove the few instances in which ChatGPT failed to produce a story and offered explanations for its inability to do so (i. e. *“As an AI language model, I cannot play games, so I can't provide a story. Would you like a different prompt?”*). Additionally, we eliminated any sections within the stories where ChatGPT referenced itself for various reasons (i. e. *“As an AI language model, I do not have personal experiences or emotions like humans do. However, here's a story for you:”*).

The file **ChatGPT_API.ipynb** uses the API of ChatGPT to generate the stories. It also shows how much money was spent on this task.

## GAN Training
To combine the **BERT and GPT-2** models we used mainly the **textrl** library but we also conducted a few experiments with the **trl** library both of which provide a set of tools for training transformers using reinforcement learning. 

The discriminator was finetuned every epoch with a dataset containing 50 good stories that were sampled from **WritingPrompts and/or ChatGPT** and 50 bad stories from the generator. The GPT2’s stories came from sampled prompts and had a maximum length of 300 tokens with the possibility of early stopping.

Normally when finetuning a network we want the model to not stray too much away from its original pretrained variant. In our case, the BERT discriminator being trained and then finetuned on proper coherent texts has a **blind spot for sequences of random tokens** which the generator could learn to reproduce to maximize reward over time. Rather than adding to the dataset of bad story sequences of random tokens and thus risking diminishing the text understanding abilities of BERT it would be a lot better to restrain the GPT2 model’s evolution. A solution for that would be incorporating into the reward the **Kullback–Leibler divergence** between a reference model and our current generator. The Kullback–Leibler divergence measures how different two probability distributions are and having it included into the reinforcement learning process would mean that the agent is discouraged from drastically changing  its policy.

Schulman et al. proposed a different way of limiting an agent’s policy updates which involves **clipping the objective between 1- ε and 1 + ε** to avoid large updates and then taking the minimum between the unclipped and clipped objectives for a lower bound overall. This approach is more memory efficient since it does not need a reference model being used to compare the probability distributions. The textrl library which we used in the experiments employs **proximal policy optimization (PPO)** algorithms to update the policy which in our case is the output of the GPT-2 model. 

Every epoch 100 prompts are sampled from WritingPrompts and for each one two different texts with a maximum of  100 tokens are generated, each text being an episode for a total of 200. Every new token is an action that is evaluated and rewarded by the BERT model based on how hard it is to distinguish the current sequence of tokens from well-written coherent stories. The reward is the output of the BERT model clipped between 0 and 1 and then shifted by -0.5 so that in the worst scenario the sequence receives a -0.5 reward and in the best 0.5. After all the episodes have been recorded, we train the generator in batches using stochastic gradient descent. Since the GPT2 model seemed to be too slow at learning from his actions we actually repeat 10 times every epoch this entire process so in total the generator learns from the experience of 2000 episodes instead of just 200.
Furthermore, the generator is finetuned every epoch with a single batch of 128 stories. This is partly because empirical evaluations have shown an increase in story coherence when following this approach. We considered that in doing so the gradient descent may reach a better optimum as it is being pushed around by two different teachers. 

The file **GAN_RL_GPT2_BERT.ipynb** is responsible for the implementation and training of the GAN model.

# Dialogue-Summarization-using-Gemma-2B
## _Check out the [Medium article](https://medium.com/@shafayet.rajit.101/summarize-dialogue-using-gemma-2b-ce5b01da283a?source=friends_link&sk=f5fcc52e3cc4420645cc30a77e4afb57) explaining the code._

In this project, I developed a dialogue summarization tool utilizing Gemma. Through using a dataset, I further refined the Large Language Model (LLM) to enhance its performance. Gemma effectively generates valuable summaries of dialogues.

## Dataset ([Kaggle](https://www.kaggle.com/datasets/marawanxmamdouh/dialogsum) | [HuggingFace](https://huggingface.co/datasets/knkarthick/dialogsum))
I have used a large-scale dialogue summarization dataset named DialogSum to fine-tune the LLM. The DialogSum dataset provides a rich source of dialogue data paired with summaries, allowing the LLM to learn the nuances of dialogue structures and content. This fine-tuning process can enhance the model's ability to summarize conversations accurately and efficiently, catering to the growing need for automated dialogue summarization in various industries. Another reason for choosing this dataset is the absence of any implementation of Gemma for dialogue summarization on this specific dataset, making this an original work. 

- Data Size: The data contains 13,460 dialogues with corresponding manually labelled summaries and topics. 
- Relevance to real-world applications: Dialogue summarization can be applied in various real-world scenarios to enhance communication, efficiency, and analysis. Some key applications include customer service chatbots, generating meeting transcriptions, creating legal transcriptions, analyzing interviews, etc. 
- Potential for creative exploration: Utilizing proper prompt engineering, we can generate customized summaries tailored to different contexts or topics. Moreover, we can focus on multiple aspects of the dialogue, such as sentiment analysis, key points extraction, and speaker attribution.  Leveraging prompt engineering to incorporate domain-specific keywords or terminologies, we can tailor the summarization output to specific industries or fields, enhancing the relevance and accuracy of generated summaries. Experimenting with prompts that encourage abstractive or extractive summarization approaches can help explore the trade-offs between generating concise summaries and preserving original dialogue content.

## Model 
I have chosen the Gemma 2B model to fine-tune based on the mentioned dataset. Gemma is a family of lightweight models that utilize the same technology as Gemini models. The reason for choosing the “2B” variant of the model is that Gemma 2B can run on mobile devices and laptops, making it efficient in computing power. 

- Relevancy: The Gemma LLM is relevant to the application of dialogue summarization because it is specifically designed to handle tasks related to understanding and processing natural language text, which includes dialogues.
- Adaptation: We can fine-tune the model using the DialogSum dataset to adapt the Gemma LLM for this dataset. This process involves training the model on the dataset to learn dialogues' specific patterns and structures and their corresponding summaries. By fine-tuning the model, we can enhance its ability to generate accurate and concise dialogue summaries, which is this application's primary goal.

## Evaluation
To determine the model's correctness, I have used the ROUGE scores to get Precision, Recall, and F1-Score of the generated summary. I have focused on ROUGE-1, ROUGE-2, ROUGE-L, and ROUGE-Lsum scores because these metrics are commonly used to evaluate the performance of summarization models in dialogue summarization tasks. ROUGE-1 measures the overlap of unigrams (individual words) between the system and reference summaries, while ROUGE-2 measures the overlap of bigrams (pairs of consecutive words). ROUGE-L, on the other hand, is based on the Longest Common Subsequence (LCS) and considers word order, which is particularly important for dialogue summarization. 

I have tested the fine-tuned model on 10 random entries of the ‘validation’ dataset. The average metrics is given below:
|  | ROUGE-1 | ROUGE-2 | ROUGE-L | ROUGE-Lsum |
| -------- | -------- | -------- | -------- | -------- |
| Precision | 0.3412 | 0.1248 | 0.2871 | 0.2871 |
| Recall | 0.3205 | 0.1205 | 0.2733 | 0.2733 |
| F1 Score | 0.3022 | 0.1057 | 0.2550 | 0.2550 |

For ROUGE-1, a Precision of around 0.34 and Recall of about 0.32 are moderate, with an F1 Score of approximately 0.30 falling within the same range. For ROUGE-2, a Precision of approximately 0.12 and Recall of around 0.12 are relatively low, resulting in an F1 Score of about 0.11, which is on the lower side. For ROUGE-L and ROUGE-Lsum, the Precision, Recall, and F1 Scores are similar to those of ROUGE-1 but slightly lower.

Gemma's underlying architecture allows it to be trained on new datasets specific to a particular task. Through further fine-tuning, these metrics can be improved. 

### Capabilities
Gemma 2B can successfully generate meaningful dialogue summaries close to the original summary. In terms of architecture, Gemma uses Multi-head Attention which allows the model to focus on specific parts of the input sequence that are most relevant to the current task. Gemma utilizes rotary positional embeddings in each layer. This helps the model understand the order of words within a sequence. During training, Gemma processes information in chunks of 8192 tokens, which gives it a decent amount of context to understand the intricacies of language. Because of these reasons, Gemma 2B can create meaningful summaries of dialogues. 

### Limitations
While Gemma's accessibility is a significant advantage, it comes with inherent trade-offs. Compared to its larger, cloud-based counterparts, Gemma's performance is inevitably limited by the processing power of a single GPU. 

Gemma models like 2B and 7B are designed to address specific computational limitations. While they offer advantages in terms of efficiency and accessibility, they may not match the power of larger AI models like OpenAI's ChatGPT-4 or Google's Gemini Ultra and Pro chatbots. 

It may only excel in some use cases, as it is optimized for specific applications rather than being a one-size-fits-all solution.

Gemma utilizes a decoder transformer architecture, limiting its functionality to text-to-text large language models. This means it may not be suitable for tasks involving images or videos that require encoder-decoder interactions like Google Gemini's multimodal capabilities.

<hr>

In conclusion, while Gemma models like 2B may not have excelled at summarizing dialogue perfectly in this task, these challenges can be addressed through further fine-tuning. Despite the inherent trade-offs and the model's focus on specific applications, I find the potential of Gemma models promising. By leveraging fine-tuning techniques and further optimization, Gemma can be tailored to excel in dialogue summarization tasks, offering efficient and accessible solutions for various industries. The underlying architecture of Gemma, with features like Multi-head Attention and rotary positional embeddings, provides a solid foundation for creating meaningful dialogue summaries. With continuous refinement and adaptation, Gemma has the potential to enhance its summarization capabilities and contribute significantly to automated dialogue summarization in real-world applications.

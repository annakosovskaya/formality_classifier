# Dataset

To ensure a fair evaluation, I decided to create my own custom dataset, rather than relying on well-known formality datasets that many language models have likely seen during pretraining. Using such public benchmarks might lead to overestimated results.

## Dataset Composition
I constructed a binary classification dataset for formality detection, combining data from six different sources:

Formal samples were collected from:
- CNN/DailyMail – news articles;

- XSum – official summaries of news;

- EURLEX – legal documents from the European Union.

Informal samples were collected from:
- TweetEval – tweets;

- Reddit – discussions and posts;

- EmpatheticDialogues – casual conversations meant to reflect everyday speech.

All datasets were loaded via the Hugging Face Hub and then cleaned and preprocessed.

## Preprocessing
To reduce noise, I applied the following preprocessing steps:

- Remove HTML artifacts (e.g. \&nbsp;, \<br>, etc.);

- Strip HTML tags;

- Normalize whitespace by removing extra spaces and tabs (formal texts usually avoid such inconsistencies, but we don't want it to be a clue for classification)

- Limit repeated characters, e.g.: "aaaaaaaaaaa" → "aaaa", "!!!!!!!!!!!!" → "!!!!!" It will not change the formality (formal texts still may have a maximum of three repeated characters).

- Filter out very short texts, which are often headlines, author names, or otherwise irrelevant.

Note: generating a dataset takes lots of memory and time. It's better to use the ready one I provided.

# _SimplifyIT_: On-Level Text Generator
## Overview
_SimplifyIT_ is (or will be) an app that evaluates and reduces the complexity of informational text so that the text can be accessed by a greater range of readers. This is particularly helpful for K-12 teachers, who are typically given instructional text that is prepared under the assumption that all of their students are able to read at the specified grade level. In practice, there is substantial hetergeneity in reading abilities in K-12 classrooms: some students may be reading below grade level or are non-native speakers sill learning the English language. Grade-level text for these students is prohibitive and can keep them from accessing content. Providing simplified text for these students will help them gain familiarity with English while also allowing them to access the material being discussed.
## Data Pipeline
The algorthims developed in this project were trained on three different data sets:
1. One Stop English Corpus 
1. Wikipedia and Simplified Wikipedia Corpus
1. WeeBit corpus
A classification machine learning algorithm is first built in order to assess the complexity of the texts. Next, a text generation model using SimpleGPT2 is fine-tuned using this text. Finally, a machine learning algorithm to evaluate the accuracy of simplified text (as compared to original text) will be built as a benchmark to evaluate the quality of the results from text generation.


```python

```

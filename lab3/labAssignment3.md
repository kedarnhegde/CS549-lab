 submit:
● Your full code
● Instructions to reproduce results
● Any dataset split information
https://huggingface.co/Dpngtm/wav2vec2-emotion-recognition
4 Dataset Description
● What dataset did you use?
Toronto Sample dataset from class
● Number of samples per emotion class
Happy: 5
Fear:  4
Angry: 4
● Train / validation / test split
● Any preprocessing performed (resampling, silence trimming, normalization)
5 Methodology
5.1 Feature Extraction
● What features were used?
 Raw waveform
 wav2vec embeddings
● Why did you choose this method?
Because we were most experienced with it following the examples from class and it fit our existing dataset file type.
5.2 Model Architecture
The model we used was pre-trained and fine-tuned from a facebook model. 
● Key hyperparameters (epochs, learning rate, batch size, etc.)
Epochs: 10
Learning Rate: 3e-5
Batch Size: 4 (Physical) / 16 (Effective via Gradient Accumulation)
6 Evaluation
6.1 Quantitative Metrics
● You must evaluate your model using quantitative metrics. Required Metrics include:
● Accuracy
● Confusion matrix
● (Optional) Precision, Recall, F1-score
6.2 Baseline Comparison
You must compare your model against the original baseline wav2vec2 model used in the class.
● Present results in a comparison table:
Model
Accuracy
Avg. Confidence
Notes
Baseline Model






Our Model








● Briefly describe:
○ Did performance improve?


○ Did confidence values increase?


○ Were predictions more stable?


6.3 Performance Analysis
Discuss:
● Which emotions were easiest to classify?
● Which emotions were most frequently confused?
● Were certain speakers harder to classify?
● How confident was the model overall?
5
7 Results & Analysis
Provide a detailed analysis of your results.
7.1 Error Analysis
Select at least three misclassified samples and:
● Show the true label vs the predicted label
● Explain why the model may have made this mistake
● Identify any confusion patterns (e.g., happy vs surprised, calm vs neutral)
7.2 Model Behavior & Robustness
Discuss:
● Did your approach improve robustness compared to the baseline?
● Did fine-tuning reduce random-like predictions?
● Did classical ML behave differently from deep learning?
● How sensitive was your model to speaker variation?
7.3 Interpretation of Results
Explain possible reasons for your observed performance:
● Dataset size limitations
● Model capacity
● Overfitting or underfitting
● Domain mismatch
● Preprocessing effects
6
8 Limitations & Future Improvements
8.1 Identified Limitations
Clearly describe at least three specific limitations of your approach. These may include:
● Limited dataset size
● Class imbalance
● Speaker variability
● Overfitting or underfitting
● Domain mismatch (training vs test data)
● Model capacity limitations
● Hardware or training constraints
For each limitation, briefly explain:
● Why does it affect performance
● How it appeared in your results
8.2 Generalization Concerns
Discuss whether your model would generalize well to:
● New speakers
● Different accents
● Different recording environments
● Real-world audio
Support your answer using evidence from your results.
7
8.3 Future Improvements
Propose at least two realistic improvements, such as:
● Using more data
● Data augmentation
● Better hyperparameter tuning
● Cross-validation
● Alternative model architectures
● Transfer learning from larger datasets
Briefly explain how each improvement could increase robustness or performance.

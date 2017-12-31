# sentiment-reviewer

I trained this neural network to determine whether a user likes or dislikes a business based on their review. To do this, I used Natural Language Processing (NLP) to process their reviews.

![network structure](https://github.com/KingArthurZ3/sentiment-reviewer/blob/master/rsc/review-dataset.png "network_structure")

## The Dataset

I used this dataset with 10,000 yelp reviews and classified them based on these 10 entries. I used Pandas, NLTK, and Scikit-learn libraries to help me analyze this dataset.

1. **business_id** (ID of the business being reviewed)
2. **date** (Day the review was posted)
3. **review_id** (ID for the posted review)
4. **stars** (1-5 rating for the business)
5. **text** (Review text)
6. **type** (Type of text)
7. **user_id** (User's id)
8. {**cool** / **useful** / **funny**} (Comments on the review, given by other users)

## Exploring the Dataset

After reading the data, I used Seaborn's FacetGrid to create histograms of my text length to the stars rating. I hoped to
identify any relationships between these two factors.

![network structure](https://github.com/KingArthurZ3/sentiment-reviewer/blob/master/rsc/review_histogram.png "network_structure")

It seems that the distribution of text is similar among all 5 ratings. However, the number of text reviews seems to be
skewed towards the 4 and 5 star reviews, which may be an issue later. Next, I created a box plot of these two categories.

![network structure](https://github.com/KingArthurZ3/sentiment-reviewer/blob/master/rsc/review_boxplot.png "network_structure")

Analyzing the plots, it appears that the 1 and 2 star reviews are much longer, but there are a lot of outliers. From this, I presumed that text length won't be as important of a feature.

Then, I decided to group the data by the star rating and tried to find a correlation between features like cool, useful,
or funny. I visualized these correlations using Seaborn's heatmap.

![network structure](https://github.com/KingArthurZ3/sentiment-reviewer/blob/master/rsc/review-heatmap.png "network_structure")

Based on the map, funny is strongly correlated with useful, and useful appears to be correlated with text length. There's
also a negative correlation between cool and the other features.

## Text Pre-processing

To classify my data, I need convert my data into some sort of feature vector. To convert my corpus to a vector of words, I used a bag-of-words approach, and represented each unique word in the text as one number. I used tools from the NLTK library to remove punctuations and stopwords, which helped me return a list of pure words, or tokens.

## Vectorization

I used Sci-kit learn's CountVectorizer to convert the my text collection into a matrix of token counts, creating a 2-D matrix, where each row is a unique word, and each column is a review.

## Training and Testing Data

I built a Multinomial Naive Bayes model, a specialised version of Naive Bayes geared towards text documents, and fit it to my training set. I tested my model by storing the predictions in a separate dataset and evaluated my predictions
against the actual ratings using matrix functions from Sci-kit learn.

## Testing and evaluating my model

After training, I compared my network's predictions against actual rating using the confusion matrix and classification reports functions from Sci-kit learn.

![network structure](https://github.com/KingArthurZ3/sentiment-reviewer/blob/master/rsc/review-results.png "network_structure")

My model achieved a **92% accuracy**, which means that it can predict whether a user liked a local business of not based on
what they typed. However, it's not without faults. 

## Data Bias

After further testing, I found that my model is more **biased** towards positive reviews in comparison to negative ones. This is most likely because my dataset had a much higher number of 5 star reviewsthan 1 star reviews, therefore it is more fitted to positive reviews.

Github Repository for this project: https://github.com/KingArthurZ3/sentiment-reviewer

## References
I obtained the dataset from this source.

**Dataset obtained from**: Kaggle: Yelp Business Rating Prediction

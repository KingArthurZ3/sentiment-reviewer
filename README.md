# sentiment-reviewer

I trained this network largely using scikit-learn libraries and visualized my training using seaborn graph helpers.

![network structure](https://github.com/KingArthurZ3/sentiment-reviewer/blob/master/rsc/review-dataset.png "network_structure")

I used this dataset with 10,000 yelp reviews and classified them based on these 10 entries.

1. **business_id** (ID of the business being reviewed)
2. **date** (Day the review was posted)
3. **review_id** (ID for the posted review)
4. **stars** (1?5 rating for the business)
5. **text** (Review text)
6. **type** (Type of text)
7. **user_id** (User?s id)
8. {**cool** / **useful** / **funny**} (Comments on the review, given by other users)

After reading the data, I used Seaborn's FacetGrid to create histograms of my text length to the stars rating. I hoped to
identify any relationships between these two factors.

![network structure](https://github.com/KingArthurZ3/sentiment-reviewer/blob/master/rsc/review_histogram.png "network_structure")

It seems that the distribution of text is similar among all 5 ratings. However, the number of text reviews seems to be
skewed towards the 4 and 5 star reviews, which may be an issue later. Next, I created a box plot of these two categories.

![network structure](https://github.com/KingArthurZ3/sentiment-reviewer/blob/master/rsc/review_boxplot.png "network_structure")

Analyzing the plots, it appears that the 1 and 2 star reviews are
much longer, but there are a lot of outliers. From this, I presumed that text length won't be as important of a feature.

Then, I decided to group the data by the star rating and tried to find a correlation between features like cool, useful,
or funny. I visualized these correlations using Seaborn's heatmap.

![network structure](https://github.com/KingArthurZ3/sentiment-reviewer/blob/master/rsc/review-heatmap.png "network_structure")

Based on the map, funny is strongly correlated with useful, and useful appears to be correlated with text length. There's
also a negative correlation between cool and the other features.

## Text Processing

I used a bag-of-words approach to convert my corpus to a vector of words, which is necessary for classifying it. I did so
by using the NLTK library to remove punctuations and stopwords.

## Training and Testing Data

I built a Multinomial Naive Bayes model, a specialised version of Naive Bayes geared towards text documents, to model and
fit my training set. I tested my model by storing the predictions in a separate dataset and evaluated my predictions
against the actual ratings using matrix functions from Sci-kit learn.

![network structure](https://github.com/KingArthurZ3/sentiment-reviewer/blob/master/rsc/review-results.png "network_structure")

My model achieved a **92% accuracy**, which means that it can predict whether a user liked a local business of not based on
what they typed. However, it's not without faults. After further testing, I found that my model is more **biased** towards
positive reviews in comparison to negative ones, most likely because my dataset had a much higher number of 5 star reviews
than 1 star reviews.
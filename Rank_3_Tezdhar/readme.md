# Innoplexus_sentiment_analysis
Codes for competition here: https://datahack.analyticsvidhya.com/contest/innoplexus-online-hiring-hackathon/

The solution here finished 2nd on public leaderboard and 3rd on private leaderboard.

### Approach:

#### Validation
 * 5 Fold stratified K-Fold as class distribution was imblanaced


#### Modeling
First impression of data suggested there were lot of wrong labels as per my perception of negative and positive sentiment.
So, I felt it would be really difficult to hand craft features and it would be best to stick to state-of-art NLP models to learn on noisy data.

I started with simple Tfidf + logistic regression model which gave me a CV score of 0.5. After looking at text data I realized there were
many rows which had lot of lines unrelated to drug. Hence, I decided to use only sentences which had drug name occuring in them.
Tfidf + logistic regression with drug name only sentences gave CV score of 0.54.

At this point, I decided to use BERT. Without any finetuning, I could only about 0.45 CV. But, once I let BERT finetune on training data,
it overfit on training data heavily but still gave a validation score of 0.60 and LB score of 0.59. Then, I added sentences which occured
before and after drug sentence, this increased CV score slightly. Then I used BERT large and finetuned it which gives CV score of 0.65 and LB 0.61.
Similarly, I finetuned XLNET base, which gave CV score 0.64 and LB 0.58. Final solution is ensemble of BERT and XLNET runs.


#### Reproducing LB results:
* Clone repo
* run `sudo apt-get install python3.6`
* run `sudo apt-get install vitualenv`
* run `mkdir data`
* copy train.csv and test.csv in data folder 
* run `bash run_all.sh`


